import json
from typing import Any, Dict, List, Optional, Union

from sqlalchemy.orm import Session

from aether.models.task.task import AetherTask


class AetherTaskCRUD:
    @staticmethod
    def create(session: Session, task: Union[AetherTask, dict]) -> AetherTask:
        if isinstance(task, dict):
            task = AetherTask(**task)

        session.add(task)
        session.commit()
        session.refresh(task)
        return task

    @staticmethod
    def get_by_id(session: Session, task_id: int) -> Optional[AetherTask]:
        return (
            session.query(AetherTask)
            .filter_by(aether_task_id=task_id, is_deleted=0)
            .first()
        )

    @staticmethod
    def list(session: Session, offset: int = 0, limit: int = 100) -> List[AetherTask]:
        return (
            session.query(AetherTask)
            .filter_by(is_deleted=0)
            .order_by(AetherTask.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    @staticmethod
    def update(session: Session, task_id: int, updates: dict) -> Optional[AetherTask]:
        task = (
            session.query(AetherTask)
            .filter_by(aether_task_id=task_id, is_deleted=0)
            .first()
        )
        if not task:
            return None
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)
        session.commit()
        session.refresh(task)
        return task

    @staticmethod
    def soft_delete(session: Session, task_id: int) -> bool:
        task = (
            session.query(AetherTask)
            .filter_by(aether_task_id=task_id, is_deleted=0)
            .first()
        )
        if not task:
            return False
        task.is_deleted = 1
        session.commit()
        return True
