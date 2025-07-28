from datetime import datetime

from pydantic import BaseModel


class TaskOut(BaseModel):
    aether_task_id: int
    task_type: str
    created_at: datetime
    updated_at: datetime
    status: int

    class Config:
        from_attributes = True
