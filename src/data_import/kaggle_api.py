from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class KaggleDatasetApi(Protocol):
    """Minimal surface used by dataset import (list files + download one file)."""

    def dataset_list_files(self, dataset_slug: str) -> list[str]: ...

    def dataset_download_file(self, dataset_slug: str, file_name: str, path: str) -> None: ...


class KaggleApiAdapter:
    """Wraps the real ``KaggleApi``; authenticates lazily. Kaggle is imported only here."""

    def __init__(self) -> None:
        self._api = None

    def _ensure_api(self) -> object:
        if self._api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi

            self._api = KaggleApi()
        return self._api

    def authenticate(self) -> None:
        self._ensure_api().authenticate()

    def dataset_list_files(self, dataset_slug: str) -> list[str]:
        api = self._ensure_api()
        result = api.dataset_list_files(dataset_slug)
        return [item.name for item in result.files]

    def dataset_download_file(self, dataset_slug: str, file_name: str, path: str) -> None:
        api = self._ensure_api()
        api.dataset_download_file(dataset_slug, file_name, path=path)
