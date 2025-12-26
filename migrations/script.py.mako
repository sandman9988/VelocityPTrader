"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

VelocityTrader Database Migration
Enterprise PostgreSQL schema management with safety checks
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}

def validate_migration():
    """Validate migration safety"""
    import os
    
    # Check environment
    env = os.getenv('TRADING_ENV', 'development')
    if env == 'production':
        print("⚠️  PRODUCTION MIGRATION")
        print("   Validate data backup before proceeding")
    
    # Check for active trading sessions
    try:
        connection = op.get_bind()
        result = connection.execute(
            "SELECT COUNT(*) FROM trading_sessions WHERE is_active = true"
        ).fetchone()
        
        if result and result[0] > 0:
            print(f"⚠️  {result[0]} active trading sessions detected")
            print("   Consider stopping trading before migration")
    except Exception:
        # Table might not exist yet
        pass

def upgrade() -> None:
    """Upgrade database schema"""
    validate_migration()
    
    print(f"📈 Upgrading to revision: ${repr(up_revision)}")
    ${upgrades if upgrades else "pass"}
    print("✅ Upgrade completed")

def downgrade() -> None:
    """Downgrade database schema"""
    validate_migration()
    
    print(f"📉 Downgrading from revision: ${repr(up_revision)}")
    ${downgrades if downgrades else "pass"}
    print("✅ Downgrade completed")