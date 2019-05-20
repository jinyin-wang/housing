import pandas as  pd
import pymysql






if __name__ == '__main__':
    df = pd.read_csv('kc_house_data.csv', encoding='gbk', usecols=[3, 4, 16])
    # df['price2'] = df['price'].map(lambda x: x / 1000)
    df['count'] = 1
    sum_bed = df['bedrooms'].groupby([df['zipcode']]).sum()
    sum_count = df['count'].groupby([df['zipcode']]).sum()

    sumall = pd.merge(sum_bed, sum_count, how='left',left_on=None, right_on=None,
      left_index=True, right_index=True)
    print(sumall)





