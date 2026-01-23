import enum
import uuid
import os
from datetime import date, datetime
from uuid import UUID as UUIDType
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import inspect
from config import URL_DATABASE
from audit import get_current_user


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


def _serialize_value(value):
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (UUIDType, uuid.UUID)):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _serialize_object(obj):
    state = inspect(obj)
    data = {}
    for attr in state.mapper.column_attrs:
        data[attr.key] = _serialize_value(getattr(obj, attr.key))
    return data


def _serialize_old_values(obj):
    state = inspect(obj)
    data = {}
    for attr in state.mapper.column_attrs:
        history = state.attrs[attr.key].history
        if history.has_changes():
            previous = None
            if history.deleted:
                previous = history.deleted[0]
            elif history.unchanged:
                previous = history.unchanged[0]
            data[attr.key] = _serialize_value(previous)
    return data


def _safe_user_id(raw_user_id):
    fallback_allowed = os.getenv("AUDIT_ALLOW_FALLBACK", "0") == "1"
    fallback = os.getenv(
        "AUDIT_FALLBACK_USER_ID", "00000000-0000-0000-0000-000000000000"
    )
    try:
        if raw_user_id:
            return uuid.UUID(str(raw_user_id))
    except (ValueError, TypeError):
        pass
    if not fallback_allowed:
        return None
    try:
        return uuid.UUID(fallback)
    except (ValueError, TypeError):
        return None


class AuditedAsyncSession(AsyncSession):
    """Async session that captures CRUD changes into audit_logs."""

    async def flush(self, *args, **kwargs):
        tracked_new = getattr(self, "_audit_created", set())
        tracked_new.update(self.new)
        self._audit_created = tracked_new
        return await super().flush(*args, **kwargs)

    async def execute(self, statement, *args, **kwargs):
        result = await super().execute(statement, *args, **kwargs)
        await self._maybe_log_read(statement)
        return result

    async def commit(self):
        audit_entries = await self._collect_audit_entries()
        await super().commit()

        if audit_entries:
            from models import AuditLog  # Lazy import to avoid circular dependency

            self.add_all([AuditLog(**entry) for entry in audit_entries])
            await super().commit()

    async def _collect_audit_entries(self):
        from models import AuditLog  # Lazy import to avoid circular dependency

        entries = []
        user_id = _safe_user_id(get_current_user())

        if user_id is None:
            return entries

        created_objs = set(getattr(self, "_audit_created", set())) | set(self.new)

        for obj in list(created_objs):
            if isinstance(obj, AuditLog):
                continue
            entries.append(
                {
                    "user_id": user_id,
                    "action": "CREATE",
                    "table_name": getattr(obj, "__tablename__", obj.__class__.__name__),
                    "old_data": None,
                    "new_data": _serialize_object(obj),
                }
            )

        for obj in list(self.dirty):
            if isinstance(obj, AuditLog):
                continue
            if not self.is_modified(obj, include_collections=False):
                continue
            entries.append(
                {
                    "user_id": user_id,
                    "action": "UPDATE",
                    "table_name": getattr(obj, "__tablename__", obj.__class__.__name__),
                    "old_data": _serialize_old_values(obj),
                    "new_data": _serialize_object(obj),
                }
            )

        for obj in list(self.deleted):
            if isinstance(obj, AuditLog):
                continue
            entries.append(
                {
                    "user_id": user_id,
                    "action": "DELETE",
                    "table_name": getattr(obj, "__tablename__", obj.__class__.__name__),
                    "old_data": _serialize_object(obj),
                    "new_data": None,
                }
            )

        return entries

    async def _maybe_log_read(self, statement):
        from sqlalchemy.sql import Select
        from sqlalchemy import insert
        from models import AuditLog

        if not isinstance(statement, Select):
            return

        user_id = _safe_user_id(get_current_user())
        if user_id is None:
            return

        table_name = None
        final_froms = statement.get_final_froms()
        if final_froms:
            table_name = (
                getattr(final_froms[0], "name", None)
                or getattr(final_froms[0], "__tablename__", None)
                or str(final_froms[0])
            )

        if not table_name:
            return

        async with self.bind.begin() as conn:
            await conn.execute(
                insert(AuditLog).values(
                    user_id=user_id,
                    action="READ",
                    table_name=table_name,
                    old_data=None,
                    new_data=None,
                )
            )


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
            class_=AuditedAsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return AsyncSessionLocal


Base = declarative_base()


async def create_tables():
    """Create all tables defined in models."""
    from models import Base as ModelsBase

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(ModelsBase.metadata.create_all)


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
