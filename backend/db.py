"""Postgres models for chats/messages via SQLAlchemy ORM — replaces the old
chat_metadata.json / chat_history.json files (one row per chat, one row per
message, instead of rewriting a whole JSON blob on every turn). No raw SQL:
all reads/writes go through the ORM session."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, ForeignKey, DateTime, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from backend.config import Config

engine = create_engine(Config.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    messages: Mapped[list["Message"]] = relationship(back_populates="chat", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chats.id", ondelete="CASCADE"))
    role: Mapped[str]
    content: Mapped[str]
    citations: Mapped[list] = mapped_column(JSONB, default=list)
    needs_clarification: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    chat: Mapped["Chat"] = relationship(back_populates="messages")


def init_schema():
    Base.metadata.create_all(engine)
