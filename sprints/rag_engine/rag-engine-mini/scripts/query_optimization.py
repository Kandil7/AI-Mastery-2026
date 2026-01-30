"""
Query Optimization Script
=============================
Identifies and suggests optimizations for slow queries.

تحسين الاستفسارات البطيئة
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text, and_, or_
from sqlalchemy.orm import sessionmaker


def check_slow_queries(limit: int = 10) -> None:
    """
    Check for slow queries using pg_stat_statements.

    Args:
        limit: Number of slowest queries to analyze
    """
    print(f"\n🐌 Checking for Slow Queries (Top {limit})...")

    engine = create_engine("postgresql+psycopg://postgres:postgres@localhost:5432/rag")

    # Get slow queries from pg_stat_statements
    slow_query_sql = text(f"""
        SELECT
            query,
            calls,
            total_time,
            mean_time,
            max_time,
            stddev_time,
            rows
        FROM pg_stat_statements
        WHERE calls > 10
        ORDER BY mean_time DESC
        LIMIT :limit
    """)

    with engine.connect() as conn:
        result = conn.execute(slow_query_sql)
        queries = result.fetchall()

    if not queries:
        print("  No queries found (database may be empty)")
        return

    # Analyze slow queries
    print(f"\n{'=' * 50}")
    print(f"  Found {len(queries)} queries analyzed")
    print(f"{'=' * 50}")

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query:")
        print(f"   Avg Time: {query.mean_time:.2f}ms")
        print(f"   Max Time: {query.max_time:.2f}ms")
        print(f"   Calls: {query.calls}")
        print(f"   Rows Affected: {query.rows}")
        print(f"   Query: {query.query[:100]}...")

        # Optimization suggestions
        if query.mean_time > 1000:  # 1 second
            print(f"   ⚠️  CRITICAL: Very slow query (> 1s)")
            print(f"      Suggestion: Add index, rewrite query, or use caching")
        elif query.mean_time > 500:  # 500ms
            print(f"   ⚠️  WARNING: Slow query (> 500ms)")
            print(f"      Suggestion: Check for full table scans, missing indexes")
        elif query.mean_time > 200:  # 200ms
            print(f"   ⚠️  NOTICE: Slower than target (> 200ms)")
            print(f"      Suggestion: Optimize query, consider caching")
        else:
            print(f"   ✅ Good performance (< 200ms)")

    # Overall statistics
    avg_time = sum(q.mean_time for q in queries) / len(queries)
    total_time = sum(q.total_time for q in queries) / 1000  # Convert to seconds
    total_calls = sum(q.calls for q in queries)

    print(f"\n{'=' * 50}")
    print("Query Performance Summary:")
    print(f"  Total Query Time: {total_time:.2f}s")
    print(f"  Total Calls: {total_calls}")
    print(f"  Average Time: {avg_time:.2f}ms")

    if avg_time > 500:
        print(f"  ⚠️  OVERALL: Queries are slow")
        print(f"      Recommendation: Review indexes, add caching, optimize schema")
    else:
        print(f"  ✅ OVERALL: Query performance is acceptable")


def suggest_indexes() -> None:
    """
    Suggest indexes based on missing foreign key indexes.

    Analyzes:
    - Tables with high query volume
    - Foreign keys without indexes
    - Common filter columns
    """
    print(f"\n🔍 Suggesting Database Indexes...")

    engine = create_engine("postgresql+psycopg://postgres:postgres@localhost:5432/rag")

    # Get table statistics
    table_stats_sql = text("""
        SELECT
            schemaname,
            tablename,
            seq_scan,
            idx_scan,
            n_tup_ins,
            n_tup_upd,
            n_tup_del
        FROM pg_stat_user_tables
        WHERE schemaname = 'public'
        ORDER BY n_tup_ins + n_tup_upd DESC
        LIMIT 10
    """)

    with engine.connect() as conn:
        result = conn.execute(table_stats_sql)
        tables = result.fetchall()

    print(f"\nTable Activity (Top 10):")
    print(f"{'-' * 50}")

    for table in tables:
        table_name = table.tablename
        seq_scan = table.seq_scan
        idx_scan = table.idx_scan
        inserts = table.n_tup_ins
        updates = table.n_tup_upd
        deletes = table.n_tup_del

        print(f"\n📊 {table_name}:")
        print(f"  Sequential Scans: {seq_scan}")
        print(f"  Index Scans: {idx_scan}")
        print(f"  Insertions: {inserts:,}")
        print(f"  Updates: {updates:,}")
        print(f"  Deletes: {deletes:,}")

        # Check for sequential scans (bad)
        total_scans = seq_scan + idx_scan
        seq_ratio = (seq_scan / total_scans * 100) if total_scans > 0 else 0

        if seq_ratio > 50:  # More than 50% sequential scans
            print(f"  ⚠️  HIGH SEQUENTIAL SCAN RATIO: {seq_ratio:.1f}%")
            print(f"      Suggestion: Add indexes, analyze WHERE clauses")
        elif seq_ratio > 20:
            print(f"  ⚠️  MODERATE SEQUENTIAL SCAN RATIO: {seq_ratio:.1f}%")
            print(f"      Suggestion: Review query plans, increase work_mem")
        else:
            print(f"  ✅ SEQUENTIAL SCAN RATIO IS GOOD: {seq_ratio:.1f}%")


def suggest_analyze() -> None:
    """
    Suggest running ANALYZE on tables.

    ANALYZE updates statistics for the query planner.
    """
    print(f"\n🔍 Suggesting ANALYZE Operations...")

    engine = create_engine("postgresql+psycopg://postgres:postgres@localhost:5432/rag")

    # Get tables that need ANALYZE
    analyze_sql = text("""
        SELECT
            schemaname,
            tablename,
            last_analyze,
            n_live_tup,
            n_dead_tup
        FROM pg_stat_all_tables
        WHERE schemaname = 'public'
        ORDER BY n_live_tup + n_dead_tup DESC
        LIMIT 10
    """)

    with engine.connect() as conn:
        result = conn.execute(analyze_sql)
        tables = result.fetchall()

    print(f"\n📊 Table Statistics (Top 10):")
    print(f"{'-' * 50}")

    for table in tables:
        print(f"\n{table.tablename}:")
        print(f"  Live Tuples: {table.n_live_tup:,}")
        print(f"  Dead Tuples: {table.n_dead_tup:,}")
        print(f"  Last ANALYZE: {table.last_analyze or 'NEVER'}")

        if not table.last_analyze or table.n_live_tup > 1000:
            print(f"  ⚠️  Recommend ANALYZE (missing or high activity)")
        else:
            print(f"  ✅ Statistics are current")

    print(f"\n💡 Recommended ANALYZE Command:")
    print(f"  ANALYZE;")
    print(f"  (Run on low-traffic periods)")


def check_vacuum() -> None:
    """
    Check for table bloat and recommend VACUUM.

    VACUUM reclaims storage and updates statistics.
    """
    print(f"\n🔍 Checking for Table Bloat (VACUUM Candidate)...")

    engine = create_engine("postgresql+psycopg://postgres:postgres@localhost:5432/rag")

    # Check for table bloat
    bloat_sql = text("""
        SELECT
            schemaname,
            tablename,
            pg_size_approx
            pg_total_relation_size
            pg_relation_size
            (pg_total_relation_size - pg_relation_size) as wasted_bytes
        FROM pg_stat_user_tables
        WHERE schemaname = 'public'
        ORDER BY (pg_total_relation_size - pg_relation_size) DESC
        LIMIT 5
    """)

    with engine.connect() as conn:
        result = conn.execute(bloat_sql)
        tables = result.fetchall()

    if not tables:
        print("  No tables found")
        return

    print(f"\n📊 Table Bloat Analysis (Top 5):")
    print(f"{'-' * 50}")

    for table in tables:
        table_name = table.tablename
        actual_size = table.pg_relation_size / 1024  # MB
        expected_size = table.pg_size_approx / 1024  # MB
        wasted_bytes = table.wasted_bytes
        wasted_mb = wasted_bytes / 1024 / 1024
        bloat_ratio = (
            (wasted_bytes / table.pg_relation_size * 100) if table.pg_relation_size > 0 else 0
        )

        print(f"\n{table_name}:")
        print(f"  Actual Size: {actual_size:.2f}MB")
        print(f"  Expected Size: {expected_size:.2f}MB")
        print(f"  Wasted Space: {wasted_mb:.2f}MB")
        print(f"  Bloat Ratio: {bloat_ratio:.1f}%")

        if bloat_ratio > 50:  # More than 50% wasted space
            print(f"  ⚠️  CRITICAL BLOAT: > 50% wasted")
            print(f"      Recommendation: VACUUM FULL {table_name};")
            print(f"      After vacuum, run: ANALYZE {table_name};")
        elif bloat_ratio > 20:
            print(f"  ⚠️  MODERATE BLOAT: > 20% wasted")
            print(f"      Recommendation: VACUUM {table_name};")
        else:
            print(f"  ✅ Bloat is acceptable")

    print(f"\n💡 Recommended VACUUM Command:")
    print(f"  VACUUM ANALYZE;")
    print(f"  (Run this during low-traffic periods)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query Optimization Script")
    parser.add_argument("--slow", action="store_true", help="Check for slow queries")
    parser.add_argument("--indexes", action="store_true", help="Suggest indexes")
    parser.add_argument("--analyze", action="store_true", help="Suggest ANALYZE")
    parser.add_argument("--vacuum", action="store_true", help="Check for table bloat")
    parser.add_argument("--all", action="store_true", help="Run all checks")

    args = parser.parse_args()

    print("=" * 60)
    print("   Database Query Optimization")
    print("=" * 60)
    print()

    if args.slow or args.all:
        check_slow_queries()

    if args.indexes or args.all:
        suggest_indexes()
        suggest_analyze()

    if args.vacuum or args.all:
        check_vacuum()

    if not any([args.slow, args.indexes, args.analyze, args.vacuum]):
        parser.print_help()
        print("\nPlease specify an option: --slow, --indexes, --analyze, --vacuum, --all")
