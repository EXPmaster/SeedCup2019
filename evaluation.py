def calculateAllMetrics(real_signed_time_array, pred_signed_time_array):
    if len(real_signed_time_array) != len(pred_signed_time_array):
        print("[Error!] in calculateAllMetrics: len(real_signed_time_array) != len(pred_signed_time_array)")
        return -1

    score_accumulate = 0
    onTime_count = 0
    total_count = len(real_signed_time_array)

    i = 0
    for item in real_signed_time_array:
        real_signed_time = item
        real_signed_time = real_signed_time.replace(minute=0)
        real_signed_time = real_signed_time.replace(second=0)
        pred_signed_time = pred_signed_time_array[i]
        time_interval = int(
            (real_signed_time - pred_signed_time).total_seconds() / 3600)

        # rankScore
        score_accumulate += time_interval**2

        # onTimePercent
        if pred_signed_time.year < 2019:
            onTime_count += 1
        elif pred_signed_time.year == 2019:
            if pred_signed_time.month < real_signed_time.month:
                onTime_count += 1
            elif pred_signed_time.month == real_signed_time.month:
                if pred_signed_time.day <= real_signed_time.day:
                    onTime_count += 1
        i += 1

    onTimePercent = float(onTime_count / total_count)
    rankScore = float((score_accumulate / total_count)**0.5)

    return onTimePercent, rankScore


def onTimePercent(pred_date, real_date):
    real_date = list(real_date)
    total = len(pred_date)
    count = 0
    for i in range(total):
        if (pred_date[i] - real_date[i]).days <= 0:
            count += 1
    print('on time percent: %lf' % (count / total))


def rankScore(real_signed_time, pred_signed_time):
    loss = []
    i = 0
    for item in real_signed_time:
        if (pred_signed_time[i] - item).days * 24 + \
                (pred_signed_time[i] - item).seconds // 3600 > 0:
            loss.append((pred_signed_time[i] - item).days * 24 +
                        (pred_signed_time[i] - item).seconds // 3600)
        else:
            loss.append((item - pred_signed_time[i]).days * 24 +
                        (item - pred_signed_time[i]).seconds // 3600)
        i += 1
    import math
    mse = math.sqrt(sum([val ** 2 for val in loss]) / len(loss))
    print('MSE: %lf' % mse)
