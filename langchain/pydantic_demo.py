from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'nisha'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=100)

# new_student = {'name': 'nisha'}

new_student = {'age': '22', 'email': 'ab@example.com', 'cgpa': 90.5}
# new_student = {}

student = Student(**new_student)

student_dict = dict(student)

print(student_dict['age'])

student_json = student.model_dump_json()

