"""index_tree: walks a tree, extracts skeletons/metadata, skips broken files."""
from lce.indexer import index_tree
from lce.retriever import Retriever


def test_indexes_fixture_tree(tmp_path):
    r = Retriever(tmp_path / "chroma")
    indexed, skipped = index_tree(r, "tests/fixtures")
    assert (indexed, skipped) == (2, 0)
    assert r.count("lean") == 2
    lean = r.query("greeting function", mode="lean", k=2)
    assert {d.doc_id for d in lean} == {"sample.py", "sample.md"}
    kinds = {d.doc_id: d.metadata["kind"] for d in lean}
    assert kinds == {"sample.py": "code", "sample.md": "doc"}


def test_skips_broken_python(tmp_path, capsys):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "ok.py").write_text('"""Ok."""\ndef f():\n    pass\n')
    (tree / "broken.py").write_text("def broken(:\n")
    r = Retriever(tmp_path / "chroma")
    indexed, skipped = index_tree(r, tree)
    assert (indexed, skipped) == (1, 1)
    assert "broken.py" in capsys.readouterr().err


def test_excludes_dot_and_artifact_dirs(tmp_path):
    tree = tmp_path / "tree"
    (tree / ".venv").mkdir(parents=True)
    (tree / ".venv" / "junk.py").write_text("def j():\n    pass\n")
    (tree / "real.py").write_text("def r():\n    pass\n")
    r = Retriever(tmp_path / "chroma")
    indexed, skipped = index_tree(r, tree)
    assert (indexed, skipped) == (1, 0)


def test_markdown_skeleton_is_lean(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "note.md").write_text(
        "# Big Title\n\nIntro text.\n\n## Alpha\n\nAlpha body.\n\n## Beta\n\nBeta body.\n"
    )
    r = Retriever(tmp_path / "chroma")
    index_tree(r, tree)
    (doc,) = r.query("alpha", mode="lean", k=1)
    assert doc.text == (
        "title: Big Title\n"
        "Big Title: Intro text.\n"
        "Alpha: Alpha body.\n"
        "Beta: Beta body."
    )
    assert "#" not in doc.text


def test_unreadable_file_is_skipped(tmp_path, capsys):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "ok.py").write_text("def f():\n    pass\n")
    secret = tree / "secret.py"
    secret.write_text("def s():\n    pass\n")
    secret.chmod(0o000)
    r = Retriever(tmp_path / "chroma")
    try:
        indexed, skipped = index_tree(r, tree)
    finally:
        secret.chmod(0o644)
    assert (indexed, skipped) == (1, 1)
    assert "secret.py" in capsys.readouterr().err


def test_constants_only_python_gets_filename_skeleton(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "consts.py").write_text("X = 1\nY = 2\n")
    r = Retriever(tmp_path / "chroma")
    indexed, skipped = index_tree(r, tree)
    assert (indexed, skipped) == (1, 0)
    (doc,) = r.query("constants", mode="lean", k=1)
    assert doc.text == "file: consts.py"
