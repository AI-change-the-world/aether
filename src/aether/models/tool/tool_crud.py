from typing import List, Optional, Union

from sqlalchemy import select
from sqlalchemy.orm import Session

from aether.models.tool.tool import AetherTool


class AetherToolCRUD:
    @staticmethod
    def create(db: Session, model_data: Union[dict, AetherTool]) -> AetherTool:
        if isinstance(model_data, dict):
            model_data = AetherTool(**model_data)
        db.add(model_data)
        db.commit()
        db.refresh(model_data)
        return model_data

    @staticmethod
    def get_by_id(db: Session, tool_id: int) -> Optional[AetherTool]:
        return (
            db.query(AetherTool)
            .filter(
                AetherTool.aether_tool_id == tool_id,
                AetherTool.is_deleted == 0,
            )
            .first()
        )

    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[AetherTool]:
        return (
            db.query(AetherTool)
            .filter(AetherTool.is_deleted == 0)
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def update(db: Session, tool_id: int, update_data: dict) -> Optional[AetherTool]:
        model = (
            db.query(AetherTool)
            .filter(
                AetherTool.aether_tool_id == tool_id,
                AetherTool.is_deleted == 0,
            )
            .first()
        )
        if not model:
            return None
        for key, value in update_data.items():
            if hasattr(model, key):
                setattr(model, key, value)
        db.commit()
        db.refresh(model)
        return model

    @staticmethod
    def delete(db: Session, tool_id: int) -> bool:
        model = (
            db.query(AetherTool)
            .filter(
                AetherTool.aether_tool_id == tool_id,
                AetherTool.is_deleted == 0,
            )
            .first()
        )
        if not model:
            return False
        model.is_deleted = 1
        db.commit()
        return True
