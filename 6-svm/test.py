import math

def total(totalYear, monthMoney, rate):
    totalMonth = totalYear * 12
    totalMoney = 0
    for i in range(totalMonth):
        totalMoney = monthMoney + totalMoney * (1 + rate/12)
        print('month: %d, total: %f' % (i, totalMoney) )
    return totalMoney

totalMoney = total(5, 100, 0.12)
print(totalMoney)
# 166722

def total3(totalYear, monthMoney, rate):
    totalMonth = totalYear * 12
    totalMoney = 0
    for i in range(totalMonth):
        thisMonthMoney = monthMoney * (1 + rate/12*(totalMonth-i))
        totalMoney += thisMonthMoney
    return totalMoney

print('2: %f' % (total3(5, 100, 0.12)))

# ＃终值=年金*（（（1+相应利率）^相应周期-1）/相应利率）

def total2(totalYear, monthMoney, rate):
    yearMoney = monthMoney * 12
    return yearMoney * (math.pow(1+rate, totalYear - 1) / rate)

t2 = total2(5, 100, 0.07)
print('', t2)

