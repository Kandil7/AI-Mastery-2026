# 数据库测试策略完全指南

## 概述

本文档提供全面的数据库测试策略，涵盖从单元测试到数据迁移验证的完整测试体系。数据库测试是确保数据完整性、应用可靠性和系统性能的关键环节，需要采用多层次、多维度的测试方法。

---

## 目录

1. [单元测试数据库操作](#1-单元测试数据库操作)
2. [集成测试与数据库](#2-集成测试与数据库)
3. [属性测试在数据库中的应用](#3-属性测试在数据库中的应用)
4. [数据库模糊测试技术](#4-数据库模糊测试技术)
5. [Schema迁移测试](#5-schema迁移测试)
6. [数据迁移验证](#6-数据迁移验证)
7. [测试最佳实践总结](#7-测试最佳实践总结)

---

## 1. 单元测试数据库操作

### 1.1 单元测试的核心原则

单元测试数据库操作的核心目标是将数据库交互与业务逻辑解耦，确保每个函数或方法在隔离环境下能够正确执行。单元测试应当快速、可靠、可重复执行，并且不依赖外部数据库实例。

#### 单元测试的关键特性

| 特性 | 描述 | 优先级 |
|------|------|--------|
| 隔离性 | 每个测试独立运行，不相互影响 | 高 |
| 快速性 | 测试执行时间应在毫秒级 | 高 |
| 可重复性 | 相同输入产生相同输出 | 高 |
| 可维护性 | 测试代码清晰易懂 | 中 |
| 覆盖率 | 确保关键路径被测试覆盖 | 高 |

### 1.2 测试替身的使用

在单元测试中，我们使用测试替身（Test Doubles）来模拟数据库行为。常见的测试替身包括：

- **Mock（模拟对象）**：预先编程的假对象，验证特定的调用
- **Stub（桩对象）**：提供预定义的响应
- **Fake（假对象）**：实现简化版本的功能
- **Spy（间谍对象）**：记录调用信息供验证

#### Python Mock 示例

```python
import unittest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

class TestUserRepository(unittest.TestCase):
    """用户仓储层的单元测试"""
    
    def setUp(self):
        """测试前置设置"""
        self.mock_connection = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_connection.cursor.return_value = self.mock_cursor
    
    def test_create_user_success(self):
        """测试创建用户成功场景"""
        # Arrange - 准备测试数据
        user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'created_at': datetime.now()
        }
        
        # 配置模拟对象
        self.mock_cursor.lastrowid = 1
        self.mock_cursor.rowcount = 1
        
        # Act - 执行被测函数
        result = self._create_user(self.mock_connection, user_data)
        
        # Assert - 验证结果
        self.assertEqual(result['id'], 1)
        self.assertTrue(result['success'])
        self.mock_cursor.execute.assert_called_once()
        self.mock_connection.commit.assert_called_once()
    
    def test_create_user_duplicate_email(self):
        """测试创建用户时邮箱重复"""
        # 配置模拟对象抛出重复键异常
        from pymysql import IntegrityError
        self.mock_cursor.execute.side_effect = IntegrityError(
            1062, "Duplicate entry 'test@example.com' for key 'email'"
        )
        
        user_data = {
            'username': 'testuser',
            'email': 'test@example.com'
        }
        
        result = self._create_user(self.mock_connection, user_data)
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'DUPLICATE_EMAIL')
    
    def test_get_user_by_id_not_found(self):
        """测试查询不存在的用户"""
        self.mock_cursor.fetchone.return_value = None
        
        result = self._get_user_by_id(self.mock_connection, 999)
        
        self.assertIsNone(result)
    
    def _create_user(self, connection, user_data):
        """实际的用户创建逻辑（被测试的目标）"""
        cursor = connection.cursor()
        try:
            cursor.execute(
                """INSERT INTO users (username, email, created_at) 
                   VALUES (%(username)s, %(email)s, %(created_at)s)""",
                user_data
            )
            connection.commit()
            return {'success': True, 'id': cursor.lastrowid}
        except Exception as e:
            connection.rollback()
            return {'success': False, 'error': str(e)}
    
    def _get_user_by_id(self, connection, user_id):
        """实际的用户查询逻辑（被测试的目标）"""
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        return cursor.fetchone()


class TestOrderService(unittest.TestCase):
    """订单服务的单元测试"""
    
    @patch('models.order.OrderRepository')
    def test_calculate_order_total(self, mock_repo_class):
        """测试订单总额计算"""
        # 模拟仓储返回订单项
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_order_items.return_value = [
            {'product_id': 1, 'quantity': 2, 'price': 100.00},
            {'product_id': 2, 'quantity': 1, 'price': 250.00},
        ]
        
        from services.order import OrderService
        service = OrderService(mock_repo)
        
        total = service.calculate_order_total(1)
        
        self.assertEqual(total, 450.00)
        mock_repo.get_order_items.assert_called_once_with(1)
```

#### Java JUnit 测试示例

```java
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
public class UserServiceTest {
    
    @Mock
    private DataSource dataSource;
    
    @Mock
    private Connection connection;
    
    @Mock
    private PreparedStatement preparedStatement;
    
    @Mock
    private ResultSet resultSet;
    
    private UserService userService;
    
    @BeforeEach
    void setUp() throws SQLException {
        when(dataSource.getConnection()).thenReturn(connection);
        when(connection.prepareStatement(anyString())).thenReturn(preparedStatement);
        when(preparedStatement.executeQuery()).thenReturn(resultSet);
        
        userService = new UserService(dataSource);
    }
    
    @Test
    void testFindUserById_Success() throws SQLException {
        // Arrange
        when(resultSet.next()).thenReturn(true);
        when(resultSet.getInt("id")).thenReturn(1);
        when(resultSet.getString("username")).thenReturn("testuser");
        when(resultSet.getString("email")).thenReturn("test@example.com");
        
        // Act
        Optional<User> result = userService.findUserById(1);
        
        // Assert
        assertTrue(result.isPresent());
        assertEquals(1, result.get().getId());
        assertEquals("testuser", result.get().getUsername());
        
        verify(preparedStatement).setInt(1, 1);
        verify(preparedStatement).executeQuery();
    }
    
    @Test
    void testFindUserById_NotFound() throws SQLException {
        // Arrange
        when(resultSet.next()).thenReturn(false);
        
        // Act
        Optional<User> result = userService.findUserById(999);
        
        // Assert
        assertFalse(result.isPresent());
    }
    
    @Test
    void testCreateUser_Success() throws SQLException {
        // Arrange
        when(preparedStatement.executeUpdate()).thenReturn(1);
        when(preparedStatement.getGeneratedKeys()).thenReturn(resultSet);
        when(resultSet.next()).thenReturn(true);
        when(resultSet.getLong(1)).thenReturn(1L);
        
        User user = new User("newuser", "new@example.com");
        
        // Act
        Long userId = userService.createUser(user);
        
        // Assert
        assertEquals(1L, userId);
        verify(connection).commit();
    }
    
    @Test
    void testCreateUser_DatabaseError() throws SQLException {
        // Arrange
        when(preparedStatement.executeUpdate()).thenThrow(
            new SQLException("Database connection failed")
        );
        
        User user = new User("newuser", "new@example.com");
        
        // Act & Assert
        assertThrows(DataAccessException.class, () -> {
            userService.createUser(user);
        });
        
        verify(connection).rollback();
    }
}
```

### 1.3 事务管理测试

事务管理是数据库操作中的关键部分，需要确保原子性、一致性、隔离性和持久性（ACID）。

```python
import unittest
from unittest.mock import MagicMock, call
from contextlib import contextmanager

class TestTransactionManagement(unittest.TestCase):
    """事务管理测试"""
    
    def setUp(self):
        self.mock_connection = MagicMock()
    
    def test_successful_transaction(self):
        """测试成功的事务提交"""
        # 创建支持上下文管理器的模拟连接
        mock_transaction = MagicMock()
        self.mock_connection.begin.return_value = mock_transaction
        
        result = self._execute_transaction(
            self.mock_connection,
            [
                ("INSERT INTO orders (id, total) VALUES (1, 100)", None),
                ("INSERT INTO order_items (order_id, product_id) VALUES (1, 1)", None),
                ("UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 1", None),
            ]
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['operations'], 3)
        self.mock_connection.commit.assert_called_once()
        self.mock_connection.rollback.assert_not_called()
    
    def test_transaction_rollback_on_error(self):
        """测试事务失败回滚"""
        mock_transaction = MagicMock()
        self.mock_connection.begin.return_value = mock_transaction
        
        # 第二次操作失败
        def execute_side_effect(sql, *args):
            if "order_items" in sql:
                raise Exception("Foreign key constraint violation")
            return MagicMock(rowcount=1)
        
        self.mock_connection.cursor.return_value.execute.side_effect = execute_side_effect
        
        result = self._execute_transaction(
            self.mock_connection,
            [
                ("INSERT INTO orders (id, total) VALUES (1, 100)", None),
                ("INSERT INTO order_items (order_id, product_id) VALUES (1, 999)", None),
            ]
        )
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], "Foreign key constraint violation")
        self.mock_connection.rollback.assert_called_once()
        self.mock_connection.commit.assert_not_called()
    
    def _execute_transaction(self, connection, operations):
        """执行事务的辅助方法"""
        cursor = connection.cursor()
        try:
            connection.begin()
            for sql, params in operations:
                cursor.execute(sql, params or ())
            connection.commit()
            return {'success': True, 'operations': len(operations)}
        except Exception as e:
            connection.rollback()
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()


class TestConcurrencyControl(unittest.TestCase):
    """并发控制测试"""
    
    def test_optimistic_locking(self):
        """测试乐观锁机制"""
        # 模拟场景：两个用户同时读取同一记录
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        
        # 第一次读取，version = 1
        mock_cursor.fetchone.return_value = {
            'id': 1,
            'name': 'Product',
            'version': 1,
            'price': 100
        }
        
        # 模拟更新时的版本检查
        update_calls = []
        def capture_update(sql, params):
            update_calls.append((sql, params))
            # 第一次更新成功，第二次失败（版本不匹配）
            if len(update_calls) > 1:
                raise Exception("Version conflict")
            return MagicMock(rowcount=1)
        
        mock_cursor.execute.side_effect = capture_update
        
        # 模拟两个并发更新
        from services.product import ProductService
        service = ProductService(mock_connection)
        
        result1 = service.update_product(1, {'price': 120}, expected_version=1)
        self.assertTrue(result1['success'])
        
        result2 = service.update_product(1, {'price': 150}, expected_version=1)
        self.assertFalse(result2['success'])
        self.assertEqual(result2['error'], 'VERSION_CONFLICT')
```

### 1.4 异常处理测试

```python
class TestDatabaseExceptionHandling(unittest.TestCase):
    """数据库异常处理测试"""
    
    def test_connection_timeout(self):
        """测试连接超时处理"""
        import socket
        
        mock_connection = MagicMock()
        mock_connection.cursor.side_effect = socket.timeout("Connection timed out")
        
        from services.user import UserService
        service = UserService(mock_connection)
        
        result = service.get_user_by_id(1)
        
        self.assertIsNone(result)
        # 验证是否正确记录了错误日志
        # mock_logger.error.assert_called()
    
    def test_deadlock_retry_logic(self):
        """测试死锁重试逻辑"""
        from pymysql import OperationalError
        
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        
        # 模拟前两次遇到死锁，第三次成功
        call_count = [0]
        def execute_with_deadlock(sql, *args):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise OperationalError(1213, "Deadlock found")
            return MagicMock(rowcount=1, fetchone={'id': 1})
        
        mock_cursor.execute.side_effect = execute_with_deadlock
        mock_connection.cursor.return_value = mock_cursor
        
        from services.transaction import TransactionService
        service = TransactionService(mock_connection)
        
        result = service.execute_with_retry(
            "UPDATE users SET last_login = NOW() WHERE id = 1",
            max_retries=3
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(call_count[0], 3)
    
    def test_sql_injection_prevention(self):
        """测试SQL注入防护"""
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        
        from services.user import UserService
        service = UserService(mock_connection)
        
        # 尝试SQL注入
        malicious_input = "'; DROP TABLE users; --"
        
        result = service.search_users(malicious_input)
        
        # 验证参数化查询被正确使用
        call_args = mock_cursor.execute.call_args
        if call_args:
            # 确保SQL语句中没有直接包含用户输入
            sql = call_args[0][0] if call_args[0] else ""
            self.assertNotIn("DROP TABLE", sql)
            self.assertNotIn("DROP", sql.upper())
```

---

## 2. 集成测试与数据库

### 2.1 集成测试架构设计

集成测试验证多个组件之间的交互，在数据库上下文中，这包括应用程序与数据库之间的交互、存储过程、触发器以及跨表的数据一致性。

#### 测试金字塔

```
           /\
          /E2E\        <- E2E测试：少量，关键用户场景
         /------\
        /Integration\  <- 集成测试：中等，覆盖组件交互
       /------------\
      /   Unit Tests \  <- 单元测试：大量，快速反馈
     /----------------\
```

### 2.2 测试数据库容器化

使用Docker容器为集成测试提供隔离、可重复的数据库环境。

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  postgres-test:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: testdb
      POSTGRES_USER: testuser
      POSTGRES_PASSWORD: testpass
    ports:
      - "5432:5432"
    volumes:
      - postgres-test-data:/var/lib/postgresql/data
      - ./test-init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U testuser -d testdb"]
      interval: 5s
      timeout: 5s
      retries: 5

  mysql-test:
    image: mysql:8.0
    environment:
      MYSQL_DATABASE: testdb
      MYSQL_USER: testuser
      MYSQL_PASSWORD: testpass
      MYSQL_ROOT_PASSWORD: rootpass
    ports:
      - "3306:3306"
    volumes:
      - mysql-test-data:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost", "-u", "root", "-prootpass"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  postgres-test-data:
  mysql-test-data:
```

```python
import pytest
import docker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import time
import os

class DatabaseTestContainer:
    """测试数据库容器管理类"""
    
    def __init__(self, db_type='postgres'):
        self.db_type = db_type
        self.client = docker.from_env()
        self.container = None
        self.engine = None
    
    def start(self):
        """启动测试数据库容器"""
        image_map = {
            'postgres': 'postgres:15-alpine',
            'mysql': 'mysql:8.0',
            'mongodb': 'mongo:6.0'
        }
        
        env_map = {
            'postgres': {
                'POSTGRES_DB': 'testdb',
                'POSTGRES_USER': 'testuser',
                'POSTGRES_PASSWORD': 'testpass'
            },
            'mysql': {
                'MYSQL_DATABASE': 'testdb',
                'MYSQL_USER': 'testuser',
                'MYSQL_PASSWORD': 'testpass',
                'MYSQL_ROOT_PASSWORD': 'rootpass'
            }
        }
        
        port_map = {
            'postgres': 5432,
            'mysql': 3306,
            'mongodb': 27017
        }
        
        self.container = self.client.containers.run(
            image_map[self.db_type],
            detach=True,
            environment=env_map[self.db_type],
            ports={f'{port_map[self.db_type]}/tcp': port_map[self.db_type]},
            remove=True,
            name=f'test-{self.db_type}-{os.getpid()}'
        )
        
        # 等待容器就绪
        self._wait_for_ready(port_map[self.db_type])
        
        # 创建SQLAlchemy引擎
        self.engine = self._create_engine(port_map[self.db_type])
        
        return self
    
    def _wait_for_ready(self, port, timeout=30):
        """等待数据库就绪"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                if result == 0:
                    time.sleep(2)  # 额外等待确保完全就绪
                    return
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError(f"Database failed to start within {timeout} seconds")
    
    def _create_engine(self, port):
        """创建SQLAlchemy引擎"""
        if self.db_type == 'postgres':
            url = 'postgresql://testuser:testpass@localhost:5432/testdb'
        elif self.db_type == 'mysql':
            url = 'mysql+pymysql://testuser:testpass@localhost:3306/testdb'
        
        engine = create_engine(url, pool_pre_ping=True)
        
        # 运行初始化SQL
        self._init_database(engine)
        
        return engine
    
    def _init_database(self, engine):
        """初始化数据库schema"""
        with engine.connect() as conn:
            # 创建测试表
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    total DECIMAL(10,2) NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.commit()
    
    def get_engine(self):
        """获取数据库引擎"""
        return self.engine
    
    def cleanup(self):
        """清理资源"""
        if self.container:
            self.container.stop(timeout=5)


@pytest.fixture(scope='session')
def postgres_db():
    """PostgreSQL测试数据库fixture"""
    with DatabaseTestContainer('postgres') as db:
        yield db.get_engine()


@pytest.fixture(scope='function')
def db_session(postgres_db):
    """每个测试函数的独立数据库会话"""
    Session = sessionmaker(bind=postgres_db)
    session = Session()
    
    # 每个测试前清理数据
    session.execute(text("TRUNCATE TABLE orders, users RESTART IDENTITY CASCADE"))
    session.commit()
    
    yield session
    
    session.close()
```

### 2.3 集成测试示例

```python
import pytest
from sqlalchemy import text
from datetime import datetime

class TestUserOrderIntegration:
    """用户订单集成测试"""
    
    def test_create_user_and_order(self, db_session):
        """测试创建用户并下单的完整流程"""
        # 1. 创建用户
        result = db_session.execute(
            text("""
                INSERT INTO users (username, email)
                VALUES (:username, :email)
                RETURNING id, username, email
            """),
            {'username': 'testuser', 'email': 'test@example.com'}
        )
        user = result.fetchone()
        user_id = user[0]
        
        # 2. 创建订单
        result = db_session.execute(
            text("""
                INSERT INTO orders (user_id, total, status)
                VALUES (:user_id, :total, :status)
                RETURNING id, user_id, total, status
            """),
            {'user_id': user_id, 'total': 299.99, 'status': 'pending'}
        )
        order = result.fetchone()
        
        db_session.commit()
        
        # 3. 验证数据完整性
        user_result = db_session.execute(
            text("SELECT id, username, email FROM users WHERE id = :id"),
            {'id': user_id}
        )
        saved_user = user_result.fetchone()
        
        order_result = db_session.execute(
            text("SELECT id, user_id, total, status FROM orders WHERE user_id = :user_id"),
            {'user_id': user_id}
        )
        saved_order = order_result.fetchone()
        
        assert saved_user[1] == 'testuser'
        assert saved_user[2] == 'test@example.com'
        assert saved_order[2] == 299.99
        assert saved_order[3] == 'pending'
    
    def test_cascade_delete_user(self, db_session):
        """测试删除用户时订单级联删除"""
        # 创建用户和订单
        user_id = db_session.execute(
            text("""
                INSERT INTO users (username, email)
                VALUES ('testuser', 'test@example.com')
                RETURNING id
            """)
        ).fetchone()[0]
        
        db_session.execute(
            text("""
                INSERT INTO orders (user_id, total)
                VALUES (:user_id, 100.00), (:user_id, 200.00)
            """),
            {'user_id': user_id}
        )
        db_session.commit()
        
        # 删除用户
        db_session.execute(
            text("DELETE FROM users WHERE id = :id"),
            {'id': user_id}
        )
        db_session.commit()
        
        # 验证订单也被删除
        order_count = db_session.execute(
            text("SELECT COUNT(*) FROM orders WHERE user_id = :user_id"),
            {'user_id': user_id}
        ).fetchone()[0]
        
        assert order_count == 0
    
    def test_foreign_key_constraint(self, db_session):
        """测试外键约束"""
        # 尝试创建不存在的用户的订单
        with pytest.raises(Exception) as exc_info:
            db_session.execute(
                text("""
                    INSERT INTO orders (user_id, total)
                    VALUES (99999, 100.00)
                """)
            )
            db_session.commit()
        
        assert 'foreign key constraint' in str(exc_info.value).lower()


class TestDatabaseConstraints:
    """数据库约束集成测试"""
    
    def test_unique_constraint_violation(self, db_session):
        """测试唯一约束"""
        # 创建第一个用户
        db_session.execute(
            text("""
                INSERT INTO users (username, email)
                VALUES ('testuser', 'test@example.com')
            """)
        )
        db_session.commit()
        
        # 尝试创建重复用户
        with pytest.raises(Exception) as exc_info:
            db_session.execute(
                text("""
                    INSERT INTO users (username, email)
                    VALUES ('testuser', 'test2@example.com')
                """)
            )
            db_session.commit()
        
        assert 'unique constraint' in str(exc_info.value).lower()
    
    def test_check_constraint(self, db_session):
        """测试检查约束"""
        # 假设orders表有CHECK (total > 0)约束
        with pytest.raises(Exception) as exc_info:
            db_session.execute(
                text("""
                    INSERT INTO orders (user_id, total)
                    VALUES (1, -100.00)
                """)
            )
            db_session.commit()
        
        assert 'check constraint' in str(exc_info.value).lower()
```

### 2.4 集成测试最佳实践

```python
# 测试数据工厂模式
class UserFactory:
    """用户测试数据工厂"""
    
    def __init__(self, db_session):
        self.db_session = db_session
    
    def create(self, **kwargs):
        """创建用户并返回"""
        default_data = {
            'username': f'user_{uuid.uuid4().hex[:8]}',
            'email': f'{uuid.uuid4().hex[:8]}@example.com'
        }
        default_data.update(kwargs)
        
        result = self.db_session.execute(
            text("""
                INSERT INTO users (username, email)
                VALUES (:username, :email)
                RETURNING id, username, email, created_at
            """),
            default_data
        )
        self.db_session.commit()
        
        row = result.fetchone()
        return {
            'id': row[0],
            'username': row[1],
            'email': row[2],
            'created_at': row[3]
        }
    
    def create_batch(self, count, **kwargs):
        """批量创建用户"""
        users = []
        for i in range(count):
            user = self.create(**kwargs)
            users.append(user)
        return users


class OrderFactory:
    """订单测试数据工厂"""
    
    def __init__(self, db_session, user_factory):
        self.db_session = db_session
        self.user_factory = user_factory
    
    def create(self, **kwargs):
        """创建订单"""
        # 确保关联用户存在
        if 'user_id' not in kwargs:
            user = self.user_factory.create()
            kwargs['user_id'] = user['id']
        
        default_data = {
            'total': 100.00,
            'status': 'pending'
        }
        default_data.update(kwargs)
        
        result = self.db_session.execute(
            text("""
                INSERT INTO orders (user_id, total, status)
                VALUES (:user_id, :total, :status)
                RETURNING id, user_id, total, status, created_at
            """),
            default_data
        )
        self.db_session.commit()
        
        row = result.fetchone()
        return {
            'id': row[0],
            'user_id': row[1],
            'total': float(row[2]),
            'status': row[3],
            'created_at': row[4]
        }


# 使用示例
class TestOrderWorkflow:
    """订单工作流集成测试"""
    
    @pytest.fixture
    def factories(self, db_session):
        """创建工厂实例"""
        user_factory = UserFactory(db_session)
        return {
            'user': user_factory,
            'order': OrderFactory(db_session, user_factory)
        }
    
    def test_complete_order_flow(self, factories):
        """测试完整的订单流程"""
        # 1. 创建用户
        user = factories['user'].create(
            username='customer1',
            email='customer1@example.com'
        )
        
        # 2. 创建多个订单
        order1 = factories['order'].create(user_id=user['id'], total=150.00)
        order2 = factories['order'].create(user_id=user['id'], total=250.00)
        
        # 3. 验证订单统计
        total = self._calculate_user_total(db_session, user['id'])
        assert total == 400.00
        
        # 4. 更新订单状态
        self._update_order_status(db_session, order1['id'], 'completed')
        
        # 5. 验证状态更新
        order = self._get_order(db_session, order1['id'])
        assert order['status'] == 'completed'
    
    def _calculate_user_total(self, db_session, user_id):
        result = db_session.execute(
            text("SELECT COALESCE(SUM(total), 0) FROM orders WHERE user_id = :user_id"),
            {'user_id': user_id}
        )
        return float(result.fetchone()[0])
    
    def _update_order_status(self, db_session, order_id, status):
        db_session.execute(
            text("UPDATE orders SET status = :status WHERE id = :id"),
            {'id': order_id, 'status': status}
        )
        db_session.commit()
    
    def _get_order(self, db_session, order_id):
        result = db_session.execute(
            text("SELECT id, user_id, total, status FROM orders WHERE id = :id"),
            {'id': order_id}
        )
        row = result.fetchone()
        return {
            'id': row[0],
            'user_id': row[1],
            'total': float(row[2]),
            'status': row[3]
        }
```

---

## 3. 属性测试在数据库中的应用

### 3.1 属性测试概念

属性测试（Property-Based Testing）是一种不同于传统示例测试的测试方法。传统测试使用具体示例验证代码行为，而属性测试通过生成大量随机输入来验证代码是否满足某些属性或不变量。

#### 属性测试的优势

| 方面 | 传统测试 | 属性测试 |
|------|----------|----------|
| 测试数量 | 手动编写有限用例 | 自动生成大量用例 |
| 边界情况 | 依赖测试人员经验 | 自动探索边界 |
| 维护成本 | 高（用例多） | 低（属性复用） |
| 发现问题 | 覆盖已知场景 | 发现未知场景 |

### 3.2 数据库属性测试框架

```python
import hypothesis
from hypothesis import given, settings, assume, example
from hypothesis import strategies as st
from datetime import datetime, timedelta
import random

# 配置Hypothesis
hypothesis.settings.register_profile(
    'database',
    max_examples=1000,
    deadline=500,
    database=None  # 禁用数据库以加速测试
)


class TestDatabaseOperationsProperties:
    """数据库操作的属性测试"""
    
    @given(
        username=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=['L', 'N'])),
        email_domain=st.domains()
    )
    @settings(max_examples=100)
    def test_user_creation_properties(self, username, email_domain):
        """测试用户创建的属性"""
        # 属性1：用户名不应为空
        assert len(username) > 0
        
        # 属性2：邮箱应该包含@
        email = f"test@{email_domain}"
        assert "@" in email
        
        # 属性3：邮箱域名应该有效
        assert "." in email_domain
    
    @given(
        users=st.lists(
            st.builds(
                lambda: {
                    'username': f"user_{random.randint(1, 10000)}",
                    'email': f"user_{random.randint(1, 10000)}@example.com"
                },
            ),
            min_size=1,
            max_size=100,
            unique_by=lambda x: x['username']
        )
    )
    def test_batch_user_creation_properties(self, users):
        """测试批量用户创建"""
        # 属性1：所有用户名应该唯一
        usernames = [u['username'] for u in users]
        assert len(usernames) == len(set(usernames))
        
        # 属性2：所有邮箱应该唯一
        emails = [u['email'] for u in users]
        assert len(emails) == len(set(emails))
        
        # 属性3：每个邮箱应该有效
        for email in emails:
            assert "@" in email
            assert "." in email.split("@")[1]


class TestTransactionProperties:
    """事务属性测试"""
    
    @given(
        initial_balance=st.floats(min_value=0, max_value=100000, allow_nan=False, allow_infinity=False),
        transaction_amount=st.floats(min_value=0.01, max_value=10000, allow_nan=False, allow_infinity=False),
        transaction_count=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=500)
    def test_balance_after_transactions(self, initial_balance, transaction_amount, transaction_count):
        """测试多次交易后的余额属性"""
        # 假设每笔交易扣除1%的手续费
        fee_rate = 0.01
        total_amount = transaction_amount * transaction_count
        total_fee = total_amount * fee_rate
        final_balance = initial_balance - total_amount - total_fee
        
        # 属性1：余额不应为负
        assert final_balance >= -10000 or initial_balance >= total_amount + total_fee
        
        # 属性2：如果初始余额充足，最终余额应该正确
        if initial_balance >= total_amount + total_fee:
            assert abs(final_balance - (initial_balance - total_amount - total_fee)) < 0.01
    
    @given(
        amounts=st.lists(
            st.floats(min_value=0.01, max_value=1000, allow_nan=False),
            min_size=1,
            max_size=50
        )
    )
    def test_order_total_calculation(self, amounts):
        """测试订单总额计算属性"""
        # 属性1：总额应为正值
        total = sum(amounts)
        assert total > 0
        
        # 属性2：总额应等于各项之和
        calculated_total = 0
        for amount in amounts:
            calculated_total += amount
        assert abs(total - calculated_total) < 0.0001
        
        # 属性3：各项金额不应超过总额
        for amount in amounts:
            assert amount <= total


class TestDataIntegrityProperties:
    """数据完整性属性测试"""
    
    @given(
        data=st.dictionaries(
            st.integers(min_value=1, max_value=10000),
            st.floats(min_value=0, max_value=1000000),
            min_size=1,
            max_size=1000
        )
    )
    def test_data_aggregation_properties(self, data):
        """测试数据聚合属性"""
        values = list(data.values())
        
        # 属性1：最小值不应超过最大值
        min_val = min(values)
        max_val = max(values)
        assert min_val <= max_val
        
        # 属性2：平均值应在最小值和最大值之间
        avg_val = sum(values) / len(values)
        assert min_val <= avg_val <= max_val
        
        # 属性3：总和应该大于等于任何单个值（对于正数）
        if min_val >= 0:
            assert sum(values) >= max_val
    
    @given(
        ids=st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=100, unique=True),
        values=st.lists(st.integers(min_value=1, max_value=10000), min_size=1, max_size=100)
    )
    def test_unique_id_mapping(self, ids, values):
        """测试唯一ID映射属性"""
        # 确保长度匹配
        min_len = min(len(ids), len(values))
        ids = ids[:min_len]
        values = values[:min_len]
        
        # 属性1：映射应该是一对一的
        mapping = dict(zip(ids, values))
        assert len(mapping) == len(ids)
        
        # 属性2：所有ID应该存在
        for id_val in ids:
            assert id_val in mapping
        
        # 属性3：所有值应该可以被访问
        for id_val in ids:
            assert mapping[id_val] is not None
```

### 3.3 数据库特定属性

```python
class TestDatabaseSchemaProperties:
    """数据库Schema属性测试"""
    
    @given(
        table_name=st.text(min_size=1, max_size=64, alphabet=st.characters(whitelist_categories=['L'], whitelist_characters='_')),
        column_name=st.text(min_size=1, max_size=64, alphabet=st.characters(whitelist_categories=['L'], whitelist_characters='_'))
    )
    def test_identifier_naming(self, table_name, column_name):
        """测试标识符命名规则"""
        # 属性1：不应以数字开头
        assert not table_name[0].isdigit()
        assert not column_name[0].isdigit()
        
        # 属性2：不应包含特殊字符（除下划线外）
        allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')
        for char in table_name:
            assert char in allowed
        for char in column_name:
            assert char in allowed
    
    @given(
        column_type=st.sampled_from(['VARCHAR', 'INTEGER', 'DECIMAL', 'TIMESTAMP', 'BOOLEAN']),
        value=st.one_of(
            st.text(),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.datetimes(),
            st.booleans()
        )
    )
    def test_type_value_compatibility(self, column_type, value):
        """测试类型值兼容性"""
        # 属性：每种类型应该能接受相应的值
        compatibility = {
            'VARCHAR': str,
            'INTEGER': int,
            'DECIMAL': (int, float),
            'TIMESTAMP': datetime,
            'BOOLEAN': bool
        }
        
        expected_type = compatibility[column_type]
        assert isinstance(value, expected_type)


class TestQueryProperties:
    """查询属性测试"""
    
    @given(
        limit=st.integers(min_value=1, max_value=1000),
        offset=st.integers(min_value=0, max_value=10000)
    )
    def test_pagination_properties(self, limit, offset):
        """测试分页属性"""
        # 属性1：limit应该为正数
        assert limit > 0
        
        # 属性2：offset应该为非负数
        assert offset >= 0
        
        # 属性3：返回的记录数不应超过limit
        total_records = 10000
        expected_count = min(limit, total_records - offset)
        assert expected_count >= 0
    
    @given(
        sort_column=st.sampled_from(['id', 'created_at', 'name', 'price']),
        sort_direction=st.sampled_from(['ASC', 'DESC'])
    )
    def test_sorting_properties(self, sort_column, sort_direction):
        """测试排序属性"""
        # 属性：排序方向应该是 ASC 或 DESC
        assert sort_direction in ['ASC', 'DESC']


class TestBusinessLogicProperties:
    """业务逻辑属性测试"""
    
    @given(
        price=st.floats(min_value=0, max_value=100000, allow_nan=False, allow_infinity=False),
        quantity=st.integers(min_value=1, max_value=1000),
        discount_rate=st.floats(min_value=0, max_value=1, allow_nan=False)
    )
    def test_discount_calculation(self, price, quantity, discount_rate):
        """测试折扣计算属性"""
        subtotal = price * quantity
        discount = subtotal * discount_rate
        total = subtotal - discount
        
        # 属性1：折扣不应超过小计
        assert discount <= subtotal
        
        # 属性2：最终金额不应为负
        assert total >= 0
        
        # 属性3：如果折扣率为0，最终金额应等于小计
        if discount_rate == 0:
            assert total == subtotal
        
        # 属性4：如果折扣率为1，最终金额应为0
        if discount_rate == 1:
            assert total == 0
    
    @given(
        principal=st.floats(min_value=100, max_value=1000000),
        annual_rate=st.floats(min_value=0.001, max_value=0.3, allow_nan=False),
        years=st.integers(min_value=1, max_value=30)
    )
    def test_interest_calculation(self, principal, annual_rate, years):
        """测试利息计算属性"""
        # 简单利息
        simple_interest = principal * annual_rate * years
        simple_total = principal + simple_interest
        
        # 复利（年复利）
        compound_total = principal * ((1 + annual_rate) ** years)
        compound_interest = compound_total - principal
        
        # 属性1：复利应该大于等于简单利息（对于正利率）
        if annual_rate > 0 and years > 1:
            assert compound_interest >= simple_interest
        
        # 属性2：最终金额应该大于本金
        assert compound_total > principal
```

---

## 4. 数据库模糊测试技术

### 4.1 模糊测试概述

数据库模糊测试（Fuzz Testing）是一种自动化测试技术，通过向系统输入随机、半随机或异常数据来发现漏洞、崩溃或其他问题。在数据库上下文中，模糊测试可以帮助发现：

- SQL注入漏洞
- 数据类型处理错误
- 边界条件问题
- 字符编码问题
- 并发控制缺陷

### 4.2 SQL模糊测试

```python
import random
import string
import sqlite3
from typing import List, Tuple
import re

class SQLFuzzer:
    """SQL模糊测试器"""
    
    def __init__(self, connection):
        self.connection = connection
        self.cursor = connection.cursor()
    
    def generate_random_string(self, length: int = 10) -> str:
        """生成随机字符串"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def generate_random_number(self, min_val: int = 0, max_val: int = 1000000) -> int:
        """生成随机数字"""
        return random.randint(min_val, max_val)
    
    def generate_malicious_input(self) -> List[str]:
        """生成恶意输入测试SQL注入"""
        return [
            "' OR '1'='1",
            "' OR '1'='1' --",
            "' OR '1'='1' /*",
            "'; DROP TABLE users; --",
            "'; DELETE FROM users; --",
            "1' AND '1'='1",
            "1' AND '1'='2",
            "1 OR 1=1",
            "1 OR 1=2",
            "' UNION SELECT * FROM users --",
            "' UNION SELECT NULL, NULL, NULL --",
            "admin' --",
            "admin' #",
            "' OR ''='",
            "' OR 'x'='x",
            "'; EXEC xp_cmdshell('dir'); --",
            "1'; WAITFOR DELAY '0:0:5' --",
        ]
    
    def generate_boundary_values(self) -> List:
        """生成边界值"""
        return [
            0,
            1,
            -1,
            127,
            128,
            255,
            256,
            32767,
            32768,
            65535,
            65536,
            2147483647,
            2147483648,
            -2147483647,
            -2147483648,
            999999999999999999999999999,
            -999999999999999999999999999,
            0.0,
            -0.0,
            float('inf'),
            float('-inf'),
            float('nan'),
        ]
    
    def generate_unicode_values(self) -> List[str]:
        """生成Unicode字符"""
        return [
            '\u0000',  # 空字符
            '\u0001',
            '\u007F',  # DEL
            '\u0080',
            '\u00FF',  # Latin-1 Supplement
            '\u0100',
            '\uFFFF',  # Non-private use High Surrogate
            '\U00010000',  # First Supplementary Character
            '\U0010FFFF',  # Maximum Unicode
            '你好世界',
            'こんにちは世界',
            '🎉🚀💻',
            '\x00\x01\x02',  # 原始字节
        ]
    
    def test_sql_injection(self, table_name: str = 'users'):
        """测试SQL注入漏洞"""
        vulnerabilities = []
        
        for malicious_input in self.generate_malicious_input():
            try:
                # 尝试在WHERE子句中使用恶意输入
                query = f"SELECT * FROM {table_name} WHERE username = '{malicious_input}'"
                self.cursor.execute(query)
                results = self.cursor.fetchall()
                
                # 如果查询成功且返回了数据，可能存在注入漏洞
                if results:
                    vulnerabilities.append({
                        'type': 'SQL_INJECTION',
                        'input': malicious_input,
                        'query': query,
                        'severity': 'HIGH'
                    })
            except Exception as e:
                # 记录错误但继续测试
                pass
        
        return vulnerabilities
    
    def test_boundary_values(self, column: str, table: str):
        """测试边界值处理"""
        issues = []
        
        for value in self.generate_boundary_values():
            try:
                query = f"SELECT * FROM {table} WHERE {column} = ?"
                self.cursor.execute(query, (value,))
                self.cursor.fetchall()
            except Exception as e:
                issues.append({
                    'type': 'BOUNDARY_ERROR',
                    'value': str(value),
                    'error': str(e),
                    'severity': 'MEDIUM'
                })
        
        return issues
    
    def test_unicode_handling(self, column: str, table: str):
        """测试Unicode字符处理"""
        issues = []
        
        for unicode_value in self.generate_unicode_values():
            try:
                # 尝试插入Unicode数据
                query = f"INSERT INTO {table} (name) VALUES (?)"
                self.cursor.execute(query, (unicode_value,))
                self.connection.commit()
                
                # 尝试读取
                self.cursor.execute(f"SELECT name FROM {table} WHERE name = ?", (unicode_value,))
                result = self.cursor.fetchone()
                
                # 验证数据一致性
                if result and result[0] != unicode_value:
                    issues.append({
                        'type': 'UNICODE_CORRUPTION',
                        'input': repr(unicode_value),
                        'output': repr(result[0]),
                        'severity': 'HIGH'
                    })
            except Exception as e:
                issues.append({
                    'type': 'UNICODE_ERROR',
                    'input': repr(unicode_value),
                    'error': str(e),
                    'severity': 'MEDIUM'
                })
        
        return issues
    
    def test_null_handling(self, table: str):
        """测试NULL值处理"""
        issues = []
        
        try:
            # 测试NULL插入
            query = f"INSERT INTO {table} (name) VALUES (NULL)"
            self.cursor.execute(query)
            self.connection.commit()
            
            # 测试NULL查询
            self.cursor.execute(f"SELECT * FROM {table} WHERE name IS NULL")
            results = self.cursor.fetchall()
            
            # 测试IS NULL vs = NULL
            self.cursor.execute(f"SELECT * FROM {table} WHERE name = NULL")
            results2 = self.cursor.fetchall()
            
            # 在标准SQL中，= NULL永远不为TRUE
            if results and not results2:
                issues.append({
                    'type': 'NULL_HANDLING_CORRECT',
                    'description': 'IS NULL works correctly, = NULL works as expected'
                })
            else:
                issues.append({
                    'type': 'NULL_HANDLING_UNEXPECTED',
                    'severity': 'LOW'
                })
        except Exception as e:
            issues.append({
                'type': 'NULL_ERROR',
                'error': str(e),
                'severity': 'LOW'
            })
        
        return issues


class DatabaseFuzzerIntegration:
    """数据库模糊测试集成"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self.fuzzer = None
    
    def setup_test_database(self):
        """设置测试数据库"""
        self.connection = sqlite3.connect(self.db_path)
        self.fuzzer = SQLFuzzer(self.connection)
        
        # 创建测试表
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                age INTEGER,
                balance REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.connection.commit()
    
    def run_all_fuzz_tests(self) -> dict:
        """运行所有模糊测试"""
        results = {
            'sql_injection': [],
            'boundary': [],
            'unicode': [],
            'null_handling': []
        }
        
        # SQL注入测试
        results['sql_injection'] = self.fuzzer.test_sql_injection('users')
        
        # 边界值测试
        results['boundary'] = self.fuzzer.test_boundary_values('age', 'users')
        
        # Unicode测试
        results['unicode'] = self.fuzzer.test_unicode_handling('username', 'users')
        
        # NULL处理测试
        results['null_handling'] = self.fuzzer.test_null_handling('users')
        
        return results
    
    def cleanup(self):
        """清理资源"""
        if self.connection:
            self.connection.close()
```

### 4.3 混沌工程与数据库

```python
import random
import time
import threading
from contextlib import contextmanager
from typing import List, Callable

class DatabaseChaosEngine:
    """数据库混沌工程引擎"""
    
    def __init__(self, connection):
        self.connection = connection
        self.active_chaos = []
    
    @contextmanager
    def network_latency(self, delay_ms: int = 1000, probability: float = 0.1):
        """模拟网络延迟"""
        chaos = {
            'type': 'NETWORK_LATENCY',
            'delay_ms': delay_ms,
            'probability': probability
        }
        self.active_chaos.append(chaos)
        
        original_execute = self.connection.cursor().execute
        
        def slow_execute(sql, *args, **kwargs):
            if random.random() < probability:
                time.sleep(delay_ms / 1000)
            return original_execute(sql, *args, **kwargs)
        
        try:
            yield chaos
        finally:
            self.active_chaos.remove(chaos)
    
    @contextmanager
    def connection_failure(self, probability: float = 0.1):
        """模拟连接失败"""
        chaos = {
            'type': 'CONNECTION_FAILURE',
            'probability': probability
        }
        self.active_chaos.append(chaos)
        
        try:
            if random.random() < probability:
                raise ConnectionError("Simulated connection failure")
            yield chaos
        finally:
            self.active_chaos.remove(chaos)
    
    @contextmanager
    def random_timeout(self, timeout_ms: int = 100, probability: float = 0.05):
        """模拟随机超时"""
        chaos = {
            'type': 'RANDOM_TIMEOUT',
            'timeout_ms': timeout_ms,
            'probability': probability
        }
        self.active_chaos.append(chaos)
        
        try:
            yield chaos
        finally:
            self.active_chaos.remove(chaos)
    
    @contextmanager
    def data_corruption(self, probability: float = 0.01):
        """模拟数据损坏"""
        chaos = {
            'type': 'DATA_CORRUPTION',
            'probability': probability
        }
        self.active_chaos.append(chaos)
        
        yield chaos
        
        self.active_chaos.remove(chaos)


class ChaosTestExample:
    """混沌工程测试示例"""
    
    def test_transaction_under_chaos(self):
        """测试混沌环境下的事务处理"""
        import sqlite3
        
        # 创建测试数据库
        conn = sqlite3.connect(':memory:')
        conn.execute("CREATE TABLE accounts (id PRIMARY KEY, balance REAL)")
        conn.execute("INSERT INTO accounts VALUES (1, 1000), (2, 1000)")
        conn.commit()
        
        chaos_engine = DatabaseChaosEngine(conn)
        
        # 在网络延迟下测试转账
        with chaos_engine.network_latency(delay_ms=100, probability=0.3):
            try:
                cursor = conn.cursor()
                cursor.execute("BEGIN TRANSACTION")
                cursor.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
                cursor.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2")
                cursor.execute("COMMIT")
                
                # 验证结果
                cursor.execute("SELECT balance FROM accounts WHERE id = 1")
                balance1 = cursor.fetchone()[0]
                
                cursor.execute("SELECT balance FROM accounts WHERE id = 2")
                balance2 = cursor.fetchone()[0]
                
                assert balance1 == 900
                assert balance2 == 1100
                
            except Exception as e:
                # 事务应该回滚
                conn.rollback()
                print(f"Transaction failed as expected: {e}")
        
        conn.close()
    
    def test_concurrent_operations_under_chaos(self):
        """测试混沌环境下的并发操作"""
        import sqlite3
        
        conn = sqlite3.connect(':memory:')
        conn.execute("CREATE TABLE counters (id PRIMARY KEY, value INTEGER)")
        conn.execute("INSERT INTO counters VALUES (1, 0)")
        conn.commit()
        
        results = []
        errors = []
        
        def increment_counter(thread_id):
            try:
                for _ in range(10):
                    cursor = conn.cursor()
                    cursor.execute("BEGIN")
                    cursor.execute("SELECT value FROM counters WHERE id = 1")
                    current = cursor.fetchone()[0]
                    time.sleep(random.random() * 0.001)
                    cursor.execute("UPDATE counters SET value = ? WHERE id = 1", (current + 1,))
                    cursor.execute("COMMIT")
                results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # 启动多个并发线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=increment_counter, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # 验证最终计数
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM counters WHERE id = 1")
        final_value = cursor.fetchone()[0]
        
        print(f"Successful threads: {len(results)}")
        print(f"Errors: {len(errors)}")
        print(f"Final counter value: {final_value}")
        
        conn.close()
```

---

## 5. Schema迁移测试

### 5.1 Schema迁移测试策略

Schema迁移是数据库演进中的关键操作，需要全面测试以确保数据完整性和应用兼容性。

```python
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
import json
import os
from datetime import datetime

class SchemaMigrationTest:
    """Schema迁移测试"""
    
    def __init__(self, source_db_url: str, target_db_url: str):
        self.source_engine = create_engine(source_db_url)
        self.target_engine = create_engine(target_db_url)
        self.source_inspector = inspect(self.source_engine)
        self.target_inspector = inspect(self.target_engine)
    
    def test_all_tables_migrated(self):
        """测试所有表都已迁移"""
        source_tables = set(self.source_inspector.get_table_names())
        target_tables = set(self.target_inspector.get_table_names())
        
        missing_tables = source_tables - target_tables
        assert len(missing_tables) == 0, f"Missing tables: {missing_tables}"
    
    def test_all_columns_preserved(self):
        """测试所有列都已保留"""
        source_tables = self.source_inspector.get_table_names()
        
        for table_name in source_tables:
            source_columns = {
                col['name']: col for col in self.source_inspector.get_columns(table_name)
            }
            target_columns = {
                col['name']: col for col in self.target_inspector.get_columns(table_name)
            }
            
            missing_columns = set(source_columns.keys()) - set(target_columns.keys())
            assert len(missing_columns) == 0, \
                f"Table {table_name} missing columns: {missing_columns}"
    
    def test_column_types_compatible(self):
        """测试列类型兼容"""
        source_tables = self.source_inspector.get_table_names()
        
        type_compatibility_map = {
            'INTEGER': ['INTEGER', 'BIGINT', 'SMALLINT'],
            'BIGINT': ['BIGINT'],
            'VARCHAR': ['VARCHAR', 'TEXT', 'CHAR'],
            'TEXT': ['TEXT', 'VARCHAR'],
            'DATE': ['DATE', 'TIMESTAMP', 'DATETIME'],
            'TIMESTAMP': ['TIMESTAMP', 'DATETIME'],
            'DECIMAL': ['DECIMAL', 'NUMERIC', 'FLOAT', 'DOUBLE'],
        }
        
        for table_name in source_tables:
            source_columns = {
                col['name']: col for col in self.source_inspector.get_columns(table_name)
            }
            target_columns = {
                col['name']: col for col in self.target_inspector.get_columns(table_name)
            }
            
            for col_name, source_col in source_columns.items():
                target_col = target_columns.get(col_name)
                if target_col:
                    source_type = source_col['type'].upper()
                    target_type = target_col['type'].upper()
                    
                    # 检查类型兼容性
                    compatible = False
                    for base_type, compatible_types in type_compatibility_map.items():
                        if base_type in source_type and target_type in compatible_types:
                            compatible = True
                            break
                    
                    assert compatible, \
                        f"Column {table_name}.{col_name} type incompatibility: \
                        {source_type} -> {target_type}"
    
    def test_primary_keys_preserved(self):
        """测试主键保留"""
        source_tables = self.source_inspector.get_table_names()
        
        for table_name in source_tables:
            source_pk = self.source_inspector.get_pk_constraint(table_name)
            target_pk = self.target_inspector.get_pk_constraint(table_name)
            
            assert set(source_pk['constrained_columns']) == set(target_pk['constrained_columns']), \
                f"Primary key mismatch for table {table_name}"
    
    def test_foreign_keys_preserved(self):
        """测试外键保留"""
        source_tables = self.source_inspector.get_table_names()
        
        for table_name in source_tables:
            source_fks = self.source_inspector.get_foreign_keys(table_name)
            target_fks = self.target_inspector.get_foreign_keys(table_name)
            
            # 简化比较（实际场景可能需要更详细的比较）
            source_fk_set = set(tuple(fk['constrained_columns']) for fk in source_fks)
            target_fk_set = set(tuple(fk['constrained_columns']) for fk in target_fks)
            
            assert source_fk_set == target_fk_set, \
                f"Foreign key mismatch for table {table_name}"
    
    def test_indexes_preserved(self):
        """测试索引保留"""
        source_tables = self.source_inspector.get_table_names()
        
        for table_name in source_tables:
            source_indexes = self.source_inspector.get_indexes(table_name)
            target_indexes = self.target_inspector.get_indexes(table_name)
            
            # 比较索引列（忽略索引名）
            source_index_cols = set(
                tuple(idx['column_names']) for idx in source_indexes if not idx['name'].startswith('sqlite_')
            )
            target_index_cols = set(
                tuple(idx['column_names']) for idx in target_indexes if not idx['name'].startswith('sqlite_')
            )
            
            assert source_index_cols == target_index_cols, \
                f"Index mismatch for table {table_name}"
    
    def test_unique_constraints_preserved(self):
        """测试唯一约束保留"""
        source_tables = self.source_inspector.get_table_names()
        
        for table_name in source_tables:
            source_unique = self.source_inspector.get_unique_constraints(table_name)
            target_unique = self.target_inspector.get_unique_constraints(table_name)
            
            # 比较唯一约束列
            source_unique_cols = set(tuple(uc['column_names']) for uc in source_unique)
            target_unique_cols = set(tuple(uc['column_names']) for uc in target_unique)
            
            assert source_unique_cols == target_unique_cols, \
                f"Unique constraint mismatch for table {table_name}"


class SchemaMigrationDataTest:
    """Schema迁移数据测试"""
    
    def __init__(self, source_db_url: str, target_db_url: str):
        self.source_engine = create_engine(source_db_url)
        self.target_engine = create_engine(target_db_url)
    
    def test_row_count_match(self):
        """测试行数匹配"""
        source_tables = inspect(self.source_engine).get_table_names()
        
        with self.source_engine.connect() as source_conn, \
             self.target_engine.connect() as target_conn:
            
            for table_name in source_tables:
                source_count = source_conn.execute(
                    text(f"SELECT COUNT(*) FROM {table_name}")
                ).scalar()
                
                target_count = target_conn.execute(
                    text(f"SELECT COUNT(*) FROM {table_name}")
                ).scalar()
                
                assert source_count == target_count, \
                    f"Row count mismatch for {table_name}: {source_count} vs {target_count}"
    
    def test_data_integrity(self):
        """测试数据完整性"""
        source_tables = inspect(self.source_engine).get_table_names()
        
        with self.source_engine.connect() as source_conn, \
             self.target_engine.connect() as target_conn:
            
            for table_name in source_tables:
                # 获取源表数据
                source_data = source_conn.execute(
                    text(f"SELECT * FROM {table_name}")
                ).fetchall()
                
                # 逐行比较
                for source_row in source_data:
                    # 构建查询条件（使用主键或所有列）
                    pk_constraint = inspect(self.source_engine).get_pk_constraint(table_name)
                    pk_columns = pk_constraint['constrained_columns']
                    
                    if pk_columns:
                        # 使用主键查询目标
                        where_clause = " AND ".join(
                            f"{col} = :{col}" for col in pk_columns
                        )
                        params = dict(zip(pk_columns, source_row[:len(pk_columns)]))
                    else:
                        # 使用所有列
                        columns = inspect(self.source_engine).get_columns(table_name)
                        where_clause = " AND ".join(
                            f"{col['name']} = :{col['name']}" for col in columns
                        )
                        params = dict(zip([col['name'] for col in columns], source_row))
                    
                    target_row = target_conn.execute(
                        text(f"SELECT * FROM {table_name} WHERE {where_clause}"),
                        params
                    ).fetchone()
                    
                    assert target_row is not None, \
                        f"Missing row in target: {table_name}, {params}"
    
    def test_null_preservation(self):
        """测试NULL值保留"""
        with self.source_engine.connect() as source_conn, \
             self.target_engine.connect() as target_conn:
            
            tables = inspect(self.source_engine).get_table_names()
            
            for table_name in tables:
                # 检查NULL值数量
                source_nulls = source_conn.execute(
                    text(f"""
                        SELECT COUNT(*) FROM {table_name} 
                        WHERE {' OR '.join(f'{col} IS NULL' for col in 
                            [c['name'] for c in inspect(self.source_engine).get_columns(table_name)])}
                    """)
                ).scalar()
                
                target_nulls = target_conn.execute(
                    text(f"""
                        SELECT COUNT(*) FROM {table_name} 
                        WHERE {' OR '.join(f'{col} IS NULL' for col in 
                            [c['name'] for c in inspect(self.target_engine).get_columns(table_name)])}
                    """)
                ).scalar()
                
                assert source_nulls == target_nulls, \
                    f"NULL count mismatch in {table_name}: {source_nulls} vs {target_nulls}"
    
    def test_unique_values_preserved(self):
        """测试唯一值保留"""
        with self.source_engine.connect() as source_conn, \
             self.target_engine.connect() as target_conn:
            
            tables = inspect(self.source_engine).get_table_names()
            
            for table_name in tables:
                # 检查每个唯一约束列的唯一值数量
                unique_constraints = inspect(self.source_engine).get_unique_constraints(table_name)
                
                for uc in unique_constraints:
                    columns = uc['column_names']
                    cols_str = ", ".join(columns)
                    
                    source_count = source_conn.execute(
                        text(f"SELECT COUNT(DISTINCT {cols_str}) FROM {table_name}")
                    ).scalar()
                    
                    target_count = target_conn.execute(
                        text(f"SELECT COUNT(DISTINCT {cols_str}) FROM {table_name}")
                    ).scalar()
                    
                    assert source_count == target_count, \
                        f"Unique value count mismatch for {table_name}.{columns}: \
                        {source_count} vs {target_count}"
```

### 5.2 回滚测试

```python
class MigrationRollbackTest:
    """迁移回滚测试"""
    
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.backup = None
    
    def backup_schema(self):
        """备份当前Schema"""
        inspector = inspect(self.engine)
        self.backup = {
            'tables': inspector.get_table_names(),
            'table_schemas': {}
        }
        
        for table_name in self.backup['tables']:
            self.backup['table_schemas'][table_name] = {
                'columns': inspector.get_columns(table_name),
                'pk': inspector.get_pk_constraint(table_name),
                'foreign_keys': inspector.get_foreign_keys(table_name),
                'indexes': inspector.get_indexes(table_name),
                'unique_constraints': inspector.get_unique_constraints(table_name)
            }
    
    def test_rollback_add_column(self):
        """测试添加列后的回滚"""
        with self.engine.connect() as conn:
            # 添加列
            conn.execute(text("ALTER TABLE users ADD COLUMN temp_column VARCHAR(100)"))
            conn.commit()
            
            # 验证列存在
            columns = [col['name'] for col in inspect(self.engine).get_columns('users')]
            assert 'temp_column' in columns
            
            # 回滚
            conn.execute(text("ALTER TABLE users DROP COLUMN temp_column"))
            conn.commit()
            
            # 验证列已删除
            columns = [col['name'] for col in inspect(self.engine).get_columns('users')]
            assert 'temp_column' not in columns
    
    def test_rollback_drop_table(self):
        """测试删除表后的回滚"""
        with self.engine.connect() as conn:
            # 创建临时表
            conn.execute(text("CREATE TABLE temp_table (id INT PRIMARY KEY, name TEXT)"))
            conn.commit()
            
            # 备份数据
            result = conn.execute(text("SELECT * FROM temp_table"))
            backup_data = result.fetchall()
            
            # 删除表
            conn.execute(text("DROP TABLE temp_table"))
            conn.commit()
            
            # 验证表已删除
            tables = inspect(self.engine).get_table_names()
            assert 'temp_table' not in tables
            
            # 重建表
            conn.execute(text("CREATE TABLE temp_table (id INT PRIMARY KEY, name TEXT)"))
            
            # 恢复数据
            for row in backup_data:
                conn.execute(
                    text("INSERT INTO temp_table (id, name) VALUES (:id, :name)"),
                    {'id': row[0], 'name': row[1]}
                )
            conn.commit()
            
            # 验证数据恢复
            result = conn.execute(text("SELECT COUNT(*) FROM temp_table"))
            assert result.scalar() == len(backup_data)
```

---

## 6. 数据迁移验证

### 6.1 数据迁移测试框架

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import hashlib

class MigrationStatus(Enum):
    """迁移状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"

@dataclass
class ValidationResult:
    """验证结果"""
    status: MigrationStatus
    total_records: int
    matched_records: int
    missing_records: int
    mismatched_records: int
    errors: List[str]
    warnings: List[str]
    
    @property
    def success_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return (self.matched_records / self.total_records) * 100
    
    @property
    def is_valid(self) -> bool:
        return self.missing_records == 0 and self.mismatched_records == 0


class DataMigrationValidator:
    """数据迁移验证器"""
    
    def __init__(self, source_db_url: str, target_db_url: str):
        self.source_engine = create_engine(source_db_url)
        self.target_engine = create_engine(target_db_url)
    
    def validate_all_tables(self) -> Dict[str, ValidationResult]:
        """验证所有表"""
        source_tables = inspect(self.source_engine).get_table_names()
        results = {}
        
        for table_name in source_tables:
            results[table_name] = self.validate_table(table_name)
        
        return results
    
    def validate_table(self, table_name: str) -> ValidationResult:
        """验证单个表"""
        errors = []
        warnings = []
        
        # 1. 检查行数
        row_count_match = self._validate_row_count(table_name)
        if not row_count_match:
            errors.append("Row count mismatch between source and target")
        
        # 2. 验证数据完整性
        missing, mismatched = self._validate_data_integrity(table_name)
        
        # 3. 验证NULL值
        nulls_match = self._validate_null_preservation(table_name)
        if not nulls_match:
            warnings.append("NULL value count differs between source and target")
        
        # 4. 验证唯一约束
        unique_match = self._validate_unique_constraints(table_name)
        if not unique_match:
            warnings.append("Unique constraint values differ")
        
        # 计算结果
        total = self._get_row_count(table_name, self.source_engine)
        matched = total - missing - mismatched
        
        status = MigrationStatus.VALIDATED if len(errors) == 0 else MigrationStatus.FAILED
        
        return ValidationResult(
            status=status,
            total_records=total,
            matched_records=matched,
            missing_records=missing,
            mismatched_records=mismatched,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_row_count(self, table_name: str) -> bool:
        """验证行数"""
        with self.source_engine.connect() as source_conn, \
             self.target_engine.connect() as target_conn:
            
            source_count = source_conn.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            ).scalar()
            
            target_count = target_conn.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            ).scalar()
            
            return source_count == target_count
    
    def _validate_data_integrity(self, table_name: str) -> Tuple[int, int]:
        """验证数据完整性"""
        missing = 0
        mismatched = 0
        
        with self.source_engine.connect() as source_conn, \
             self.target_engine.connect() as target_conn:
            
            # 获取主键列
            pk_constraint = inspect(self.source_engine).get_pk_constraint(table_name)
            pk_columns = pk_constraint['constrained_columns']
            
            if not pk_columns:
                warnings.append(f"Table {table_name} has no primary key, skipping detailed validation")
                return 0, 0
            
            # 获取所有行
            source_data = source_conn.execute(
                text(f"SELECT * FROM {table_name}")
            ).fetchall()
            
            source_columns = [col['name'] for col in inspect(self.source_engine).get_columns(table_name)]
            
            for row in source_data:
                # 构建主键条件
                pk_values = {col: row[source_columns.index(col)] for col in pk_columns}
                where_clause = " AND ".join(f"{k} = :{k}" for k in pk_values.keys())
                
                # 查询目标表
                target_row = target_conn.execute(
                    text(f"SELECT * FROM {table_name} WHERE {where_clause}"),
                    pk_values
                ).fetchone()
                
                if target_row is None:
                    missing += 1
                elif not self._compare_rows(row, target_row, source_columns):
                    mismatched += 1
            
            return missing, mismatched
    
    def _compare_rows(self, source_row: tuple, target_row: tuple, columns: List[str]) -> bool:
        """比较两行数据"""
        for i, col in enumerate(columns):
            source_val = source_row[i]
            target_val = target_row[i]
            
            # 处理特殊类型比较
            if isinstance(source_val, (int, float)) and isinstance(target_val, (int, float)):
                if abs(source_val - target_val) > 0.0001:
                    return False
            elif source_val != target_val:
                return False
        
        return True
    
    def _validate_null_preservation(self, table_name: str) -> bool:
        """验证NULL值保留"""
        columns = [col['name'] for col in inspect(self.source_engine).get_columns(table_name)]
        
        with self.source_engine.connect() as source_conn, \
             self.target_engine.connect() as target_conn:
            
            for col in columns:
                source_nulls = source_conn.execute(
                    text(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL")
                ).scalar()
                
                target_nulls = target_conn.execute(
                    text(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL")
                ).scalar()
                
                if source_nulls != target_nulls:
                    return False
        
        return True
    
    def _validate_unique_constraints(self, table_name: str) -> bool:
        """验证唯一约束"""
        unique_constraints = inspect(self.source_engine).get_unique_constraints(table_name)
        
        with self.source_engine.connect() as source_conn, \
             self.target_engine.connect() as target_conn:
            
            for uc in unique_constraints:
                cols = ", ".join(uc['column_names'])
                
                source_count = source_conn.execute(
                    text(f"SELECT COUNT(DISTINCT {cols}) FROM {table_name}")
                ).scalar()
                
                target_count = target_conn.execute(
                    text(f"SELECT COUNT(DISTINCT {cols}) FROM {table_name}")
                ).scalar()
                
                if source_count != target_count:
                    return False
        
        return True
    
    def _get_row_count(self, table_name: str, engine) -> int:
        """获取行数"""
        with engine.connect() as conn:
            return conn.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            ).scalar()


class DataChecksumValidator:
    """数据校验和验证器"""
    
    def __init__(self, source_db_url: str, target_db_url: str):
        self.source_engine = create_engine(source_db_url)
        self.target_engine = create_engine(target_db_url)
    
    def calculate_table_checksum(self, table_name: str, engine) -> str:
        """计算表的校验和"""
        with engine.connect() as conn:
            # 获取所有列
            columns = [col['name'] for col in inspect(engine).get_columns(table_name)]
            
            # 构建校验和查询
            checksum_query = text(f"""
                SELECT MD5(GROUP_CONCAT(
                    CONCAT({', '.join(f"COALESCE(CAST({col} AS CHAR), '')" for col in columns)})
                ORDER BY {columns[0]}))
                AS checksum
                FROM {table_name}
            """)
            
            result = conn.execute(checksum_query)
            return result.fetchone()[0]
    
    def validate_checksum(self, table_name: str) -> bool:
        """验证表的数据校验和"""
        source_checksum = self.calculate_table_checksum(table_name, self.source_engine)
        target_checksum = self.calculate_table_checksum(table_name, self.target_engine)
        
        return source_checksum == target_checksum
    
    def validate_all_checksums(self) -> Dict[str, bool]:
        """验证所有表的校验和"""
        source_tables = inspect(self.source_engine).get_table_names()
        results = {}
        
        for table_name in source_tables:
            results[table_name] = self.validate_checksum(table_name)
        
        return results
```

### 6.2 数据质量验证

```python
class DataQualityValidator:
    """数据质量验证器"""
    
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
    
    def validate_data_types(self, table_name: str) -> List[Dict]:
        """验证数据类型"""
        issues = []
        columns = inspect(self.engine).get_columns(table_name)
        
        with self.engine.connect() as conn:
            for col in columns:
                col_name = col['name']
                expected_type = str(col['type'])
                
                # 采样查询
                result = conn.execute(
                    text(f"SELECT {col_name} FROM {table_name} LIMIT 100")
                )
                
                for row in result:
                    value = row[0]
                    if value is not None:
                        if not self._check_type_compatibility(value, expected_type):
                            issues.append({
                                'table': table_name,
                                'column': col_name,
                                'value': str(value)[:50],
                                'expected_type': expected_type,
                                'issue': 'Type mismatch'
                            })
                            break
        
        return issues
    
    def _check_type_compatibility(self, value: Any, expected_type: str) -> bool:
        """检查类型兼容性"""
        type_upper = expected_type.upper()
        
        if 'INT' in type_upper:
            return isinstance(value, int)
        elif 'VARCHAR' in type_upper or 'TEXT' in type_upper:
            return isinstance(value, str)
        elif 'DECIMAL' in type_upper or 'NUMERIC' in type_upper or 'FLOAT' in type_upper or 'DOUBLE' in type_upper:
            return isinstance(value, (int, float))
        elif 'DATE' in type_upper or 'TIMESTAMP' in type_upper:
            return isinstance(value, datetime)
        elif 'BOOL' in type_upper:
            return isinstance(value, bool)
        
        return True
    
    def validate_referential_integrity(self, table_name: str) -> List[Dict]:
        """验证引用完整性"""
        issues = []
        foreign_keys = inspect(self.engine).get_foreign_keys(table_name)
        
        with self.engine.connect() as conn:
            for fk in foreign_keys:
                columns = fk['constrained_columns']
                referred_table = fk['referred_table']
                referred_columns = fk['referred_columns']
                
                for col, ref_col in zip(columns, referred_columns):
                    # 查找孤立记录
                    result = conn.execute(text(f"""
                        SELECT COUNT(*) 
                        FROM {table_name} t
                        WHERE t.{col} IS NOT NULL
                        AND NOT EXISTS (
                            SELECT 1 FROM {referred_table} r
                            WHERE r.{ref_col} = t.{col}
                        )
                    """))
                    
                    orphan_count = result.scalar()
                    
                    if orphan_count > 0:
                        issues.append({
                            'table': table_name,
                            'column': col,
                            'referred_table': referred_table,
                            'orphan_count': orphan_count,
                            'issue': 'Referential integrity violation'
                        })
        
        return issues
    
    def validate_business_rules(self, table_name: str, rules: List[Dict]) -> List[Dict]:
        """验证业务规则"""
        issues = []
        
        with self.engine.connect() as conn:
            for rule in rules:
                rule_name = rule['name']
                condition = rule['condition']
                
                result = conn.execute(text(f"""
                    SELECT COUNT(*) 
                    FROM {table_name}
                    WHERE {condition}
                """))
                
                violation_count = result.scalar()
                
                if violation_count > 0:
                    issues.append({
                        'table': table_name,
                        'rule': rule_name,
                        'condition': condition,
                        'violation_count': violation_count,
                        'issue': 'Business rule violation'
                    })
        
        return issues
    
    def generate_data_quality_report(self, table_name: str) -> Dict:
        """生成数据质量报告"""
        report = {
            'table': table_name,
            'total_rows': 0,
            'null_counts': {},
            'duplicate_counts': {},
            'data_type_issues': [],
            'referential_issues': [],
            'quality_score': 100.0
        }
        
        with self.engine.connect() as conn:
            # 总行数
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            report['total_rows'] = result.scalar()
            
            # NULL值统计
            columns = inspect(self.engine).get_columns(table_name)
            for col in columns:
                col_name = col['name']
                result = conn.execute(
                    text(f"SELECT COUNT(*) FROM {table_name} WHERE {col_name} IS NULL")
                )
                null_count = result.scalar()
                if null_count > 0:
                    report['null_counts'][col_name] = null_count
                    report['quality_score'] -= (null_count / report['total_rows']) * 10
            
            # 重复值统计
            for col in columns:
                col_name = col['name']
                result = conn.execute(text(f"""
                    SELECT {col_name}, COUNT(*) as cnt
                    FROM {table_name}
                    GROUP BY {col_name}
                    HAVING COUNT(*) > 1
                """))
                duplicates = result.fetchall()
                if duplicates:
                    report['duplicate_counts'][col_name] = len(duplicates)
                    report['quality_score'] -= 5
        
        # 数据类型问题
        report['data_type_issues'] = self.validate_data_types(table_name)
        if report['data_type_issues']:
            report['quality_score'] -= 20
        
        # 引用完整性问题
        report['referential_issues'] = self.validate_referential_integrity(table_name)
        if report['referential_issues']:
            report['quality_score'] -= 30
        
        report['quality_score'] = max(0, report['quality_score'])
        
        return report
```

---

## 7. 测试最佳实践总结

### 7.1 测试金字塔实践

```
                    /\
                   /E2E\           <- 5-10% 测试数量
                  /------\         关键用户场景
                 /Integration\     <- 15-25% 测试数量
                /   Service    \   组件交互
               /    Tests       \   
              /------------------\
             /     Unit Tests    \  <- 70-80% 测试数量
            /   (Database Mocks)  \  快速反馈
           /______________________\
```

### 7.2 测试数据管理策略

| 策略 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| 内存数据库 | 单元测试、CI | 快速、隔离 | 功能受限 |
| Docker容器 | 集成测试 | 真实环境 | 启动慢 |
| 数据库快照 | 回归测试 | 可重复 | 占用空间 |
| 测试数据工厂 | 动态测试 | 灵活 | 需维护 |

### 7.3 测试覆盖矩阵

| 测试类型 | 覆盖内容 | 自动化程度 |
|----------|----------|------------|
| 单元测试 | 业务逻辑、边界条件 | 高 |
| 集成测试 | 数据一致性、约束 | 高 |
| 属性测试 | 随机场景、边界 | 高 |
| 模糊测试 | 异常输入、安全 | 中 |
| 迁移测试 | 数据完整性 | 高 |
| 性能测试 | 响应时间、吞吐量 | 中 |

### 7.4 持续集成配置

```yaml
# .github/workflows/database-tests.yml
name: Database Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install pytest pytest-cov pytest-mock
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov --cov-report=xml

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: testdb
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Run integration tests
        run: |
          pytest tests/integration/ -v

  property-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run property-based tests
        run: |
          pytest tests/property/ -v --hypothesis-show-statistics
```

---

## 附录：常用数据库测试命令

### PostgreSQL

```bash
# 查看表结构
\d users

# 查看索引
\d index_name

# 查看外键
\df

# 执行EXPLAIN ANALYZE
EXPLAIN ANALYZE SELECT * FROM users WHERE id = 1;

# 查看锁信息
SELECT * FROM pg_locks;

# 查看活动连接
SELECT * FROM pg_stat_activity;

# 查看慢查询
SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;
```

### MySQL

```sql
-- 查看表结构
DESCRIBE users;
SHOW CREATE TABLE users;

-- 查看索引
SHOW INDEX FROM users;

-- 执行EXPLAIN
EXPLAIN SELECT * FROM users WHERE id = 1;

-- 查看进程列表
SHOW PROCESSLIST;

-- 查看 InnoDB 状态
SHOW ENGINE INNODB STATUS;

-- 查看慢查询日志
SHOW VARIABLES LIKE 'slow_query_log';
SELECT * FROM mysql.slow_log;
```

---

*本文档是数据库测试策略的完整指南，涵盖从单元测试到数据迁移验证的各个层面。建议结合项目实际情况选择合适的测试策略组合。*
