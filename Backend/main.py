from fastapi import Depends, FastAPI, HTTPException, status, File, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from typing import Annotated
from datetime import timedelta
import login
import os

from database import SessionLocal, engine
import crud
import models
import schemas

models.Base.metadata.create_all(bind=engine)

app = FastAPI()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(token: Annotated[str, Depends(login.oauth2_scheme)], db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, login.SECRET_KEY, algorithms=[login.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = schemas.UserBase(email=email)
    except JWTError:
        raise credentials_exception
    user = crud.get_user_by_email(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user


def get_current_active_user(
        current_user: Annotated[schemas.User, Depends(get_current_user)]
):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token", response_model=schemas.Token)
def login_for_access_token(
        form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
        db: Session = Depends(get_db)
):
    print(form_data.username, form_data.password)
    user = crud.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=login.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = login.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=schemas.User)
def read_users_me(current_user: Annotated[schemas.User, Depends(get_current_active_user)]):
    return current_user


@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)


@app.get("/users/", response_model=list[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@app.post("/uploadfile/")
def create_upload_file(file: UploadFile = File(...)):
    file_location = f'./uploads/{file.filename}'
    with open(file_location, 'wb+') as file_object:
        file_object.write(file.file.read())
    return {'info': f"file '{file.filename}' uploaded at '{file_location}'"}


@app.get("/listfiles/")
def list_files():
    upload_folder = "./uploads/"
    files = os.listdir(upload_folder)
    return {'files': files}


@app.post("/sources/")
def create_source(source: schemas.SourceCreate, db: Session = Depends(get_db)):
    db_source = crud.get_source_by_name(db, source_name=source.name)
    if db_source:
        raise HTTPException(status_code=400, detail="Source already exists")
    return crud.create_source(db=db, source=source)


@app.get("/source/{source_name}", response_model=schemas.Source)
def read_source(source_name: str, db: Session = Depends(get_db)):
    db_source = crud.get_source_by_name(db, source_name=source_name)
    if db_source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return db_source


@app.get("/sources/", response_model=list[schemas.Source])
def read_sources(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    sources = crud.get_sources(db, skip=skip, limit=limit)
    return sources


@app.post("/detections/")
def create_detection(detection: schemas.DetectionCreate, db: Session = Depends(get_db)):
    return crud.create_detection(db=db, detection=detection)


@app.get("/detection_search/", response_model=list[schemas.Detection])
def read_detections_filtered(source_id: int = None, county: str = None, identifier: str = None,
                             datetime1: str = None, datetime2: str = None, db: Session = Depends(get_db)):
    return crud.get_detections_by_filter(db, source_id=source_id, county=county, identifier=identifier,
                                         datetime1=datetime1, datetime2=datetime2)


@app.get("/detections/", response_model=list[schemas.Detection])
def read_detections(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    detections = crud.get_detections(db, skip=skip, limit=limit)
    return detections
