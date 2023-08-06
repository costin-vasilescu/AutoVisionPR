from sqlalchemy.orm import Session
import models
import schemas
import login


def get_source(db: Session, source_id: int):
    return db.query(models.Source).filter(models.Source.id == source_id).first()


def get_source_by_name(db: Session, source_name: str):
    return db.query(models.Source).filter(models.Source.name == source_name).first()


def get_sources(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Source).offset(skip).limit(limit).all()


def get_detection(db: Session, detection_id: int):
    return db.query(models.Detection).filter(models.Detection.id == detection_id).first()


def get_detections(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Detection).offset(skip).limit(limit).all()


def get_detections_by_filter(db: Session, datetime1: str = None, datetime2: str = None, **kwargs):
    filters = []
    for key, value in kwargs.items():
        if value:
            if key not in ['identifier']:
                filters.append(getattr(models.Detection, key) == value)
            else:
                filters.append(models.Detection.identifier.contains(kwargs['identifier']))

    if datetime1 and datetime2:
        filters.append(models.Detection.timestamp.between(datetime1, datetime2))

    return db.query(models.Detection).filter(*filters).all()


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_source(db: Session, source: schemas.SourceCreate):
    db_source = models.Source(
        name=source.name,
        type=source.type,
        length=source.length,
        detections=source.detections
    )
    db.add(db_source)
    db.commit()
    db.refresh(db_source)
    return db_source


def create_detection(db: Session, detection: schemas.DetectionCreate):
    db_detection = models.Detection(
        source_id=detection.source_id,
        frame_id=detection.frame_id,
        tracking_id=detection.tracking_id,
        bndbox=detection.bndbox,
        plate_confidence=detection.plate_confidence,
        text_score=detection.text_score,
        county=detection.county,
        identifier=detection.identifier,
        timestamp=detection.timestamp,
    )
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)
    return db_detection


def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = login.get_password_hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        return False
    if not login.verify_password(password, user.hashed_password):
        return False
    return user
