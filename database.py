from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from config import URL_DATABASE


# Convert PostgreSQL URL from psycopg2 format to asyncpg format
# Example: postgresql:// -> postgresql+asyncpg://
def get_async_db_url(url):
    """
    Converts a PostgreSQL URL from psycopg2 format to asyncpg format.

    This function takes a database URL and converts it to the format required by asyncpg
    if it's a PostgreSQL URL. If the URL already uses the asyncpg format or is for a
    different database type, it's returned unchanged.

    Args:
        url: The database URL string to convert

    Returns:
        The converted URL with the asyncpg driver prefix if applicable, otherwise the original URL
    """
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://")
    return url


async_db_url = get_async_db_url(URL_DATABASE)

# Global variable to store the engine
async_engine = None


async def initialize_database():
    """
    Initializes the database by checking if it exists and creating the engine.
    This should be called at application startup.
    """
    global async_engine

    # Create async engine
    async_engine = create_async_engine(async_db_url, echo=False, future=True)

    return async_engine


def get_engine():
    """
    Returns the async engine. If not initialized, raises an error.
    """
    if async_engine is None:
        raise RuntimeError(
            "Database not initialized. Call initialize_database() first."
        )
    return async_engine


# Create async session factory - will be initialized after engine creation
AsyncSessionLocal = None


def get_session_factory():
    """
    Returns the async session factory. Creates it if not already initialized.
    """
    global AsyncSessionLocal
    if AsyncSessionLocal is None:
        engine = get_engine()
        AsyncSessionLocal = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return AsyncSessionLocal


Base = declarative_base()


# Dependency to get async DB session
async def get_db():
    """
    Dependency function to get an async database session.

    This function creates and yields an async database session using the AsyncSessionLocal
    factory. It ensures that the session is properly closed after use, even if an exception
    occurs during the request handling.

    Yields:
        AsyncSession: An async SQLAlchemy session for database operations
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
