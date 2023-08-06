from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float
from database import Base


class Source(Base):
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    type = Column(String, index=True)
    length = Column(Float)
    detections = Column(Integer)


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey('sources.id'), index=True)
    frame_id = Column(Integer, index=True)
    tracking_id = Column(Integer)
    bndbox = Column(String)
    plate_confidence = Column(Float)
    text_score = Column(Float)
    county = Column(String, index=True)
    identifier = Column(String)
    timestamp = Column(String, index=True)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
