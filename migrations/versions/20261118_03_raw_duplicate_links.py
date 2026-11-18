"""Stage 2: link exact copies of articles without a promoted News row.

Revision ID: 20261118_03
Revises: 20261118_02
This additive migration never deletes or overwrites historical raw/news data.
"""
from alembic import op
import sqlalchemy as sa

revision = "20261118_03"
down_revision = "20261118_02"
branch_labels = None
depends_on = None


def upgrade():
    inspector = sa.inspect(op.get_bind())
    if "news_intake_records" not in inspector.get_table_names():
        raise RuntimeError("Run the preceding 20261118_02 migration first")
    columns = {col["name"] for col in inspector.get_columns("news_intake_records")}
    if "duplicate_of_intake_id" in columns:
        raise RuntimeError("Column already present: inspect alembic_version, do not overwrite")
    op.add_column(
        "news_intake_records",
        sa.Column("duplicate_of_intake_id", sa.Integer(),
                  sa.ForeignKey("news_intake_records.id"), nullable=True),
    )
    op.create_index("ix_intake_duplicate_raw", "news_intake_records",
                    ["duplicate_of_intake_id"])


def downgrade():
    raise RuntimeError("Destructive downgrade disabled: raw duplicate links must be retained.")
