

from flask import Flask, render_template
import pymysql

app = Flask(__name__)


class Database:
    def __init__(self):
        host = "127.0.0.1"
        user = "root"
        password = "wangjinyin521"
        db = "mysql"
        self.con = pymysql.connect(host=host, user=user, password=password, db=db, cursorclass=pymysql.cursors.
                                   DictCursor)
        self.cur = self.con.cursor()

    def list_employees(self):
        self.cur.execute("SELECT first_name, last_name, gender FROM employees LIMIT 50")
        result = self.cur.fetchall()

        return result

    def get_prediction(self, ame):
        sql = "SELECT first_name, last_name, gender FROM employees where first_name='%s' LIMIT 50" % ame
        print(sql)
        self.cur.execute(sql)
        result = self.cur.fetchall()
        print(result)
        return result

@app.route('/')
def employees():
    def db_query():
        db = Database()
        emps = db.list_employees()
        print(emps)
        return emps

    res = db_query()

    return render_template('employees.html', result=res, content_type='application/json')


@app.route('/limit')
def one_employees():
    def db_query_name():
        db = Database()
        emps = db.get_prediction('joey')
        print(emps)
        return emps

    res = db_query_name()

    return render_template('employees.html', result=res, content_type='application/json')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=81)