# 导入 MySQL 连接库
import pymysql
# 导入pandas
import pandas as pd
import sys, os

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f'current_dir--》{current_dir}')
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
print(f'rag_qa_path--》{rag_qa_path}')
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
print(f'project_root--》{project_root}')
sys.path.insert(0, project_root)

# 导入配置和日志
from base import Config, logger


class MySQLClient:
    def __init__(self):
        # 初始化日志
        self.logger = logger
        # 连接参数只解析一次，重连时复用（不必每次重连都重新读 config.ini）
        self._conf = Config()
        try:
            # 连接 MySQL 数据库（启动阶段连不上就直接失败）
            self._connect()
            # 记录连接成功
            self.logger.info("MySQL 连接成功")
        except pymysql.MySQLError as e:
            # 记录连接失败
            self.logger.error(f"MySQL 连接失败: {e}")
            raise

    def _connect(self):
        """建立一条新的 MySQL 连接，并创建绑定在它上面的游标"""
        self._connection = pymysql.connect(
            host=self._conf.MYSQL_HOST,
            user=self._conf.MYSQL_USER,
            password=self._conf.MYSQL_PASSWORD,
            database=self._conf.MYSQL_DATABASE,
            charset="utf8mb4",
            # port=3307
        )
        self._cursor = self._connection.cursor()

    def _ensure_alive(self):
        """取用前探活并自动重连。

        MySQL 服务端的 wait_timeout（本项目是 120 秒）会把空闲长连接直接踢掉，
        之后再 execute 就会抛 pymysql.err.OperationalError(0, '')，
        表现为“获取对话历史失败: (0, '')”“更新会话历史失败: (0, '')”。
        这里在每次取连接/游标前 ping 一次：pymysql 会在同一个 Connection 对象上
        原地重连，原有游标依然可用，所以上层 `cursor.execute(...)` 的写法无需改动。
        """
        try:
            self._connection.ping(reconnect=True)
        except Exception as e:
            # 连 ping 都救不回来（服务重启、网络切换等）→ 彻底重建连接和游标
            self.logger.warning(f"MySQL 连接已失效，重建连接: {e}")
            try:
                self._connection.close()
            except Exception:
                pass
            self._connect()

    @property
    def connection(self):
        """保活后的连接对象，上层可直接 commit()/rollback()"""
        self._ensure_alive()
        return self._connection

    @property
    def cursor(self):
        """保活后的游标对象，上层可直接 execute()/fetchall()"""
        self._ensure_alive()
        return self._cursor

    def create_table(self):
        create_table_query = '''
                             CREATE TABLE IF NOT EXISTS jpkb
                             (
                                 id
                                 INT
                                 AUTO_INCREMENT
                                 PRIMARY
                                 KEY,
                                 subject_name
                                 VARCHAR
                             (
                                 20
                             ),
                                 question VARCHAR
                             (
                                 1000
                             ),
                                 answer VARCHAR
                             (
                                 1000
                             )
                                 ) \
                             '''
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            self.logger.info("表创建成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"表创建失败: {e}")
            raise

    def insert_data(self, csv_path):
        try:
            data = pd.read_csv(csv_path)
            # 数据单条写入
            for _, row in data.iterrows():
                insert_query = "INSERT INTO jpkb (subject_name, question, answer) VALUES (%s, %s, %s)"
                self.cursor.execute(insert_query, (row['学科名称'], row['问题'], row['答案']))
            self.connection.commit()
            self.logger.info("数据插入成功")
        except Exception as e:
            self.logger.error(f"数据插入失败: {e}")
            self.connection.rollback()  # 数据的回滚
            raise

    def fetch_questions(self):
        # 获取所有问题
        try:
            # 执行查询
            self.cursor.execute("SELECT question FROM jpkb")
            # 获取结果
            results = self.cursor.fetchall()
            # 记录获取成功
            self.logger.info("成功获取问题")
            # 返回结果
            return results
        except pymysql.MySQLError as e:
            # 记录查询失败
            self.logger.error(f"查询失败: {e}")
            # 返回空列表
            return []

    def fetch_answer(self, question):
        # 获取指定问题的答案
        try:
            # 执行查询
            self.cursor.execute("SELECT answer FROM jpkb WHERE question=%s", (question,))
            # 获取结果
            result = self.cursor.fetchone()
            # 返回答案或 None
            return result[0] if result else None
        except pymysql.MySQLError as e:
            # 记录答案获取失败
            self.logger.error(f"答案获取失败: {e}")
            # 返回 None
            return None

    def close(self):
        # 关闭数据库连接
        try:
            # 直接操作底层对象，避免经过 property 触发一次无谓的保活重连
            if getattr(self, "_cursor", None) is not None:
                self._cursor.close()
            if getattr(self, "_connection", None) is not None:
                self._connection.close()
            # 记录关闭成功
            self.logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            # 记录关闭失败
            self.logger.error(f"关闭连接失败: {e}")


if __name__ == '__main__':
    mysql_client = MySQLClient()
    mysql_client.create_table()
    mysql_client.insert_data('../data/JP学科知识问答.csv')
