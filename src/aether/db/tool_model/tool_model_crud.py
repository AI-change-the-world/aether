from typing import List, Optional, Union

from sqlalchemy import select
from sqlalchemy.orm import Session

from aether.db.tool_model.tool_model import AetherToolModel


class AetherToolModelCRUD:
    @staticmethod
    def create(
        db: Session, model_data: Union[dict, AetherToolModel]
    ) -> AetherToolModel:
        if isinstance(model_data, dict):
            model_data = AetherToolModel(**model_data)
        db.add(model_data)
        db.commit()
        db.refresh(model_data)
        return model_data

    @staticmethod
    def get_by_id(db: Session, model_id: int) -> Optional[AetherToolModel]:
        return (
            db.query(AetherToolModel)
            .filter(
                AetherToolModel.aether_tool_model_id == model_id,
                AetherToolModel.is_deleted == 0,
            )
            .first()
        )

    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[AetherToolModel]:
        return (
            db.query(AetherToolModel)
            .filter(AetherToolModel.is_deleted == 0)
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def update(
        db: Session, model_id: int, update_data: dict
    ) -> Optional[AetherToolModel]:
        model = (
            db.query(AetherToolModel)
            .filter(
                AetherToolModel.aether_tool_model_id == model_id,
                AetherToolModel.is_deleted == 0,
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
    def delete(db: Session, model_id: int) -> bool:
        model = (
            db.query(AetherToolModel)
            .filter(
                AetherToolModel.aether_tool_model_id == model_id,
                AetherToolModel.is_deleted == 0,
            )
            .first()
        )
        if not model:
            return False
        model.is_deleted = 1
        db.commit()
        return True
