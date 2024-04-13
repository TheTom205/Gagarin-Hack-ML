from fastapi import FastAPI
from model import find_courses, find_friends
from pydantic import BaseModel


class UserData(BaseModel):
    description: str | None = None

app = FastAPI()


@app.post("/getCourses")
async def getCourses(userData: UserData):
    return find_courses(userData.description)

@app.post("/getFriends")
async def getFriends(userData: UserData):
    return find_friends(userData.description)
