"""Command line: ``fridge2fork <command>``."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__, synthetic
from .config import Settings
from .data import RecipeSchemaError, load_recipes
from .evaluate import evaluate
from .explain import grounded_explanation, template_explanation
from .rank import ALLERGENS
from .recommender import Recommender, result_to_dict


def _llm(s: Settings):
    if s.llm == "openai":
        from .llm import OpenAIChat  # noqa: PLC0415

        return OpenAIChat(s.llm_model, s.llm_base_url, s.llm_timeout_s)
    return None


def cmd_synth(args, s):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    synthetic.generate(args.n, seed=s.seed).to_csv(out, index=False)
    print(f"wrote {args.n} synthetic recipes to {out}")


def cmd_build(args, s):
    path = Path(args.recipes or s.recipes_path)
    recipes, rep = load_recipes(path)
    print("\n".join(rep.lines()))
    rec = Recommender.build(recipes, s)
    folder = rec.save(args.index_dir or s.index_dir, source=path)
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    print(f"index: {m['recipes']} recipes, {m['vocabulary']} ingredients, {m['embedded_ingredients']} embedded, "
          f"{m['recipes_without_vector']} recipes without a vector -> {folder}")


def cmd_recommend(args, s):
    rec = Recommender.load(args.index_dir or s.index_dir, s)
    res = rec.recommend(args.ingredients, k=args.k, max_missing=args.max_missing,
                        exclude_allergens=tuple(args.exclude or ()), vegetarian=args.vegetarian)
    if args.json:
        print(json.dumps(result_to_dict(res), indent=2, default=str))
        return
    for w in res.warnings:
        print(f"note: {w}")
    print(f"known ingredients: {', '.join(res.query.known) or '-'} | candidates: {res.candidates}")
    llm = _llm(s) if args.explain else None
    for n, r in enumerate(res.recommendations, 1):
        print(f"\n{n}. {r.name}  score {r.score:.3f}  coverage {r.coverage:.0%}  missing {len(r.missing)}"
              + (f"  cosine {r.dense:.3f}" if r.dense is not None else ""))
        print(f"   matched: {', '.join(r.matched) or '-'}")
        print(f"   missing: {', '.join(r.missing) or '-'}")
        if r.allergens:
            print(f"   allergens (keyword check): {', '.join(r.allergens)}")
        if args.explain:
            exp = grounded_explanation(r, res.query.known, llm) if llm else template_explanation(r)
            print("   " + exp.text.replace("\n", "\n   "))
            if exp.note:
                print(f"   ({exp.note})")


def cmd_evaluate(args, s):
    path = Path(args.recipes or s.recipes_path)
    recipes, _ = load_recipes(path)
    table = evaluate(recipes, s, n_queries=args.queries, hide=args.hide)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)


def cmd_ui(args, s):  # pragma: no cover - starts a server
    app = Path(__file__).with_name("app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def cmd_demo(args, s):
    out = Path(args.out_dir) if args.out_dir else Path("artifacts/demo")
    data = out / "synthetic_recipes.csv"
    cmd_synth(argparse.Namespace(out=data, n=1500), s)
    cmd_build(argparse.Namespace(recipes=data, index_dir=out / "index"), s)
    print("\n== query: 'tomatoes, pasta, garlic, basil, mozarella' ==")
    cmd_recommend(argparse.Namespace(index_dir=out / "index", ingredients="tomatoes, pasta, garlic, basil, mozarella",
                                     k=3, max_missing=None, exclude=None, vegetarian=False, json=False,
                                     explain=True), replace(s, llm="none"))
    print("\n== offline evaluation (hide 2 ingredients, 300 queries) ==")
    cmd_evaluate(argparse.Namespace(recipes=data, queries=300, hide=2, out=out / "evaluation.csv"), s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fridge2fork", description="Recipes from the ingredients that you have.")
    p.add_argument("--version", action="version", version=f"fridge2fork {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="write a synthetic recipe file")
    s.add_argument("--out", default="data/synthetic_recipes.csv")
    s.add_argument("-n", type=int, default=1500)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("build", help="normalise the recipes, fit the embeddings and save the index")
    s.add_argument("--recipes")
    s.add_argument("--index-dir")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("recommend", help="recommend recipes for a list of ingredients")
    s.add_argument("ingredients", help='comma-separated, for example "tomato, pasta, garlic"')
    s.add_argument("--index-dir")
    s.add_argument("-k", type=int)
    s.add_argument("--max-missing", type=int)
    s.add_argument("--exclude", action="append", choices=sorted(ALLERGENS), help="allergen group to exclude")
    s.add_argument("--vegetarian", action="store_true")
    s.add_argument("--explain", action="store_true", help="explanation with the real steps")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_recommend)

    s = sub.add_parser("evaluate", help="leave-k-out recall@k and MRR")
    s.add_argument("--recipes")
    s.add_argument("--queries", type=int, default=300)
    s.add_argument("--hide", type=int, default=2)
    s.add_argument("--out")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("ui", help="start the Streamlit app (extra ui)")
    s.set_defaults(func=cmd_ui)

    s = sub.add_parser("demo", help="offline demo on synthetic recipes")
    s.add_argument("--out-dir")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rc = args.func(args, Settings.from_env())
    except (RecipeSchemaError, FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
