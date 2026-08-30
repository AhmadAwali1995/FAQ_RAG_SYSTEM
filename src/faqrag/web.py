"""Optional browser chat UI, served by the API.

This module and the ``web/`` directory are the *entire* UI. Nothing in the
retrieval or generation pipeline imports them, and :mod:`faqrag.api` mounts them
through a guarded import, so the system degrades to an API-only service if they
are absent.

To remove the UI permanently::

    rm -rf web/ src/faqrag/web.py

To disable it without deleting anything::

    FAQRAG_ENABLE_CHAT_UI=false

The page is plain HTML/CSS/JS with no build step and no CDN, so it works offline
and needs no toolchain. It calls the same ``/query``, ``/retrieve``, and
``/health`` endpoints any other client would, on the same origin -- which is
also why there is no CORS configuration to get wrong.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.html"


def build_chat_router(web_dir: Path) -> APIRouter:
    """Build the router that serves the chat UI from ``web_dir``.

    Args:
        web_dir: Directory containing ``index.html``.

    Returns:
        A router exposing ``GET /`` (redirect) and ``GET /chat``.
    """
    router = APIRouter(tags=["chat-ui"])
    index_path = web_dir / INDEX_FILENAME

    @router.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Send the bare host to the chat UI."""
        return RedirectResponse(url="/chat")

    @router.get("/chat", include_in_schema=False)
    def chat() -> FileResponse:
        """Serve the chat UI page."""
        if not index_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"chat UI not found at {index_path}; set FAQRAG_ENABLE_CHAT_UI=false",
            )
        # no-cache keeps edits to index.html visible on reload during development.
        return FileResponse(
            index_path, media_type="text/html", headers={"Cache-Control": "no-cache"}
        )

    return router


def mount_chat_ui(app: FastAPI, settings) -> bool:
    """Attach the chat UI to ``app`` if it is enabled and present.

    Never raises: a missing or disabled UI is a normal state, not an error, and
    must not stop the API from serving.

    Args:
        app: The FastAPI application to mount onto.
        settings: The active :class:`~faqrag.config.Settings`.

    Returns:
        ``True`` if the UI was mounted.
    """
    if not settings.enable_chat_ui:
        logger.info("chat UI disabled by configuration")
        return False

    web_dir = Path(settings.web_dir)
    if not (web_dir / INDEX_FILENAME).is_file():
        logger.warning("chat UI enabled but %s is missing; serving API only", web_dir)
        return False

    app.include_router(build_chat_router(web_dir))
    logger.info("chat UI available at /chat")
    return True
