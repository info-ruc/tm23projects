"""empty message

Revision ID: 8b5b7a145ce7
Revises: 10570f800d11
Create Date: 2019-11-16 20:41:03.403563

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '8b5b7a145ce7'
down_revision = '10570f800d11'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('article', sa.Column('tags', sa.String(length=100), nullable=False))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('article', 'tags')
    # ### end Alembic commands ###