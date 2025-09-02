"""Main entry point for the SD3 Large API server."""

import logging
import uvicorn

from src.sd3_api.api import app
from src.sd3_api.config import HOST, PORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Starting SD3 Large API server...")
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )