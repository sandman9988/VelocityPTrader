"""
Database migration environment for VelocityTrader
Enterprise PostgreSQL migrations with safety checks
"""

import logging
import os
from logging.config import fileConfig
from urllib.parse import quote_plus

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Import models for autogenerate
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.database.models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

def get_database_url():
    """Get database URL from environment or config"""
    # Try environment variables first (for different environments)
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    database = os.getenv('POSTGRES_DB', 'velocity_trader')
    username = os.getenv('POSTGRES_USER', 'velocity_trader')
    password = os.getenv('POSTGRES_PASSWORD', '')
    
    password_part = f":{quote_plus(password)}" if password else ""
    
    url = f"postgresql://{username}{password_part}@{host}:{port}/{database}"
    
    # Override with config if set
    config_url = config.get_main_option("sqlalchemy.url")
    if config_url and not config_url.startswith("postgresql://velocity_trader:@"):
        return config_url
    
    return url

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_schemas=False,
        # Custom migration options
        transaction_per_migration=True,
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()

def validate_migration_safety():
    """Validate that migration is safe to run"""
    # Check if we're in production
    env = os.getenv('TRADING_ENV', 'development')
    if env == 'production':
        print("⚠️  PRODUCTION MIGRATION DETECTED")
        print("   Ensure you have:")
        print("   - Database backup")
        print("   - Maintenance window")
        print("   - Rollback plan")
        
        # In production, require explicit confirmation
        confirmation = os.getenv('CONFIRM_PRODUCTION_MIGRATION', 'false')
        if confirmation.lower() != 'true':
            raise RuntimeError(
                "Production migration requires CONFIRM_PRODUCTION_MIGRATION=true"
            )

def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Validate migration safety
    validate_migration_safety()
    
    # Configure engine
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_database_url()
    
    # Additional PostgreSQL optimizations for migrations
    configuration["sqlalchemy.pool_pre_ping"] = "true"
    configuration["sqlalchemy.pool_recycle"] = "3600"
    configuration["sqlalchemy.echo"] = os.getenv('MIGRATION_ECHO_SQL', 'false').lower() == 'true'
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args={
            "options": "-c timezone=UTC",
            "application_name": "VelocityTrader_Migration"
        }
    )

    with connectable.connect() as connection:
        # Set PostgreSQL session parameters
        connection.execute("SET statement_timeout = '300000'")  # 5 minutes
        connection.execute("SET lock_timeout = '30000'")       # 30 seconds
        connection.execute("SET timezone = 'UTC'")
        
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_schemas=False,
            transaction_per_migration=True,
            # Custom options
            render_as_batch=False,
            # Include constraint names for better diff detection
            include_name=True,
            # Migration naming convention
            naming_convention={
                "ix": "ix_%(column_0_label)s",
                "uq": "uq_%(table_name)s_%(column_0_name)s",
                "ck": "ck_%(table_name)s_%(constraint_name)s",
                "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
                "pk": "pk_%(table_name)s"
            }
        )

        with context.begin_transaction():
            # Log migration start
            revision = context.get_revision_argument()
            print(f"🔄 Running migration: {revision}")
            print(f"   Database: {configuration['sqlalchemy.url'].split('@')[1] if '@' in configuration['sqlalchemy.url'] else 'local'}")
            print(f"   Environment: {os.getenv('TRADING_ENV', 'development')}")
            
            try:
                context.run_migrations()
                print(f"✅ Migration completed successfully: {revision}")
            except Exception as e:
                print(f"❌ Migration failed: {revision}")
                print(f"   Error: {e}")
                raise

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()