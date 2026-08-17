import json

from rag_pdf.config import settings

# Pastas vazias não têm chunk nenhum no Chroma pra guardar o metadado "folder",
# então esse registro à parte é o que permite criar/manter uma pasta antes de
# qualquer arquivo ser indexado nela.


def _registry_path():
    return settings.vectorstore_path / "folders.json"


def _load() -> set[str]:
    path = _registry_path()
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


def _save(folders: set[str]) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(folders), ensure_ascii=False, indent=2), encoding="utf-8")


def list_folders() -> list[str]:
    return sorted(_load())


def create_folder(folder: str) -> None:
    if not folder:
        return
    folders = _load()
    folders.add(folder)
    _save(folders)


def delete_folder(folder: str) -> None:
    folders = _load()
    remaining = {f for f in folders if f != folder and not f.startswith(folder + "/")}
    _save(remaining)


def rename_folder(old_folder: str, new_folder: str) -> None:
    folders = _load()
    updated = set()
    for f in folders:
        if f == old_folder:
            updated.add(new_folder)
        elif f.startswith(old_folder + "/"):
            updated.add(new_folder + f[len(old_folder) :])
        else:
            updated.add(f)
    updated.add(new_folder)
    _save(updated)
