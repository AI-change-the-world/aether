from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from aether.db.task.task import AetherTask


def get_task(db: Session, aether_task_id: int) -> AetherTask:
    return (
        db.query(AetherTask)
        .filter_by(aether_task_id=aether_task_id, is_deleted=0)
        .first()
    )


def update_task(
    db: Session, aether_task_id: int, updates: Dict[str, Any]
) -> Optional[AetherTask]:
    task = (
        db.query(AetherTask)
        .filter_by(aether_task_id=aether_task_id, is_deleted=0)
        .first()
    )
    if not task:
        return None

    for key, value in updates.items():
        if hasattr(task, key):
            setattr(task, key, value)

    db.commit()
    db.refresh(task)
    return task
