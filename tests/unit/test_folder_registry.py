from unittest.mock import patch

from rag_pdf.vectorstore import folder_registry


def test_create_and_list_folder(tmp_path):
    with patch("rag_pdf.vectorstore.folder_registry.settings") as mock_settings:
        mock_settings.vectorstore_path = tmp_path
        folder_registry.create_folder("Suporte/2024")
        assert folder_registry.list_folders() == ["Suporte/2024"]


def test_delete_folder_removes_subfolders(tmp_path):
    with patch("rag_pdf.vectorstore.folder_registry.settings") as mock_settings:
        mock_settings.vectorstore_path = tmp_path
        folder_registry.create_folder("Suporte")
        folder_registry.create_folder("Suporte/2024")
        folder_registry.create_folder("Manuais")

        folder_registry.delete_folder("Suporte")

        assert folder_registry.list_folders() == ["Manuais"]


def test_rename_folder_updates_subfolders(tmp_path):
    with patch("rag_pdf.vectorstore.folder_registry.settings") as mock_settings:
        mock_settings.vectorstore_path = tmp_path
        folder_registry.create_folder("Suporte")
        folder_registry.create_folder("Suporte/2024")

        folder_registry.rename_folder("Suporte", "Produtos")

        assert folder_registry.list_folders() == ["Produtos", "Produtos/2024"]


def test_create_folder_ignores_empty_string(tmp_path):
    with patch("rag_pdf.vectorstore.folder_registry.settings") as mock_settings:
        mock_settings.vectorstore_path = tmp_path
        folder_registry.create_folder("")
        assert folder_registry.list_folders() == []
