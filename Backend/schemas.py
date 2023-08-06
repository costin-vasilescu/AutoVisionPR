from pydantic import BaseModel


# class ItemBase(BaseModel):
#     title: str
#     description: str | None = None
#
#
# class ItemCreate(ItemBase):
#     pass
#
#
# class Item(ItemBase):
#     id: int
#     owner_id: int
#
#     class Config:
#         orm_mode = True

class SourceBase(BaseModel):
    name: str
    type: str
    length: float | None = None
    detections: int


class SourceCreate(SourceBase):
    pass


class Source(SourceBase):
    id: int

    class Config:
        orm_mode = True


class DetectionBase(BaseModel):
    source_id: int
    frame_id: int
    tracking_id: int | None = None
    bndbox: str
    plate_confidence: float | None = None
    text_score: float | None = None
    county: str | None = None
    identifier: str | None = None
    timestamp: str


class DetectionCreate(DetectionBase):
    pass


class Detection(DetectionBase):
    id: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    is_active: bool

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str
