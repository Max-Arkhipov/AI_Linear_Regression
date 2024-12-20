# AI_Linear_Regression
HW project I ( 2024, ML course).

## Результаты работы

### Предобработка данных и EDA

- В первую очередь был осуществлен первичный анализ данных: вывели различные части датасета, определили типы полей и наличие в них пропусков, вывели список полей с пропусками. Кроме того, с помощью библиотеки ydata_profiling создали отчет, вывели его на экран, сохрнили в файл profile_report.html.

- Далее была проведена обработка полей mileage, engine, max_power, torque: очистили их от единиц измерения и привели к числовым типам. При реализации были написаны функции с использованием методов обработки строк. 

- Признак torque я сначала удалил, но после решил попробовать построить модель с ним. Метрики качества увеличились, но незначительно.

- После заполнили пропуски в полях тренировочного и тестового датасета медианными значениями по столбцам тренировочного датасета. При реализации использовались функции и циклы.

- Удалили из тренировочного датасета объекты с одинаковым признаковым описанием, обновили индексы строк.

- Построили графики `pairplot` из библиотеки `seabron` для тренировочного и тестового датасета. Были сделаны следующие выводы:
    - На основе распределений можно предположить связь признаков с целевой переменной, в особенности сильную связь можно наблюдать для признаков: year, engine, max_power и km_driven. Связь признаков engine и max_power с целевой переменной скорее всего линейная. Зависимость между признаками year, km_driven и целевой переменной выглядит квадратичной.
    - На основе распределений можно выдвинуть гипотезу о корреляциях признаков: engine, mileage, max_power, seats.
      
- Построили тепловую карту `heatmap` из библиотеки `seabron` для отображения корреляций тренировочного датасета. Из тепловой карты следует, что наибольшая зависимость наблюдается между таргетом и признаком max_power. На тепловой карте подтверждается зависимость между признаками engine - seats, engine - max_power.

- Для того, чтобы сравнить распределения тестового и тренировочного набора данных построены дополнительные графики отображающие данные на одном графическом элементе, кроме того в анализ включены категориальные признаки. Некоторые выводы:
    - Данные на гистограмме позволяют предположить схожесть распределения признаков датасетов, однако для распределения автомобилей по году производства можно заметить, что в тренировочном датасете не представлены 2005, 2009 и 2015 годы выпуска автомобилей.
    - Данные, представленные на графике `kdeplot` библиотеки `seabron`, показывают высокую схожесть распределения признаков датасета.
    - При отображении на тепловой карте всех признаков модели можно заметить сильную корреляцию между наименованием марки автомобиля и таргетом. В дальнейшем при построении модели было принято решение обработать столбец с маркой автомобиля и использовать его для предсказания цены, что позволило значительно улучшить качество модели.

### Модель только на вещественных признаках

- Обучили классическую линейную регрессию. Получили результат:
    - начение MSE для трейна:  116583825077.70851
    - Значение MSE для теста:   232880347004.4386
    - Значение R^2 для трейна:  0.593272045373965
    - Значение R^2 для теста:   0.5948699056090114
    - Модель далека от совершенства, предсказывает цену посредственно.

- Стандартизировали классическую линейную регрессию. Получили следующий результат:
    - Значение MSE для трейна:  116583825077.70865
    - Значение MSE для теста:   232880347004.43484
    - Значение R^2 для трейна:  0.5932720453739646
    - Значение R^2 для теста:   0.594869905609018
    - Каких-либо значимых изменений качества модели стандартизация не дала
    - Наиболее информативным оказался признак max_power.
    - Проведена стандартизация признаков, последующие модели на вещественных признаках обучаются на них.
      
- Обучили Lasso-регрессию:
    - Значение MSE для трейна:  116583825087.82166
    - Значение MSE для теста:   232881031337.54358
    - Значение R^2 для трейна:  0.5932720453386833
    - Значение R^2 для теста:   0.5948687151095168
    - Качество модели практически не изменилось
    - L1-регуляризация с параметрами по умолчанию не занулила никакие веса

- Перебором по сетке (c 10-ю фолдами) подобрали оптимальные параметры для Lasso-регрессии:
    - Получили: alpha = 55000
    - Значение MSE для трейна:  124451265802.738
    - Значение MSE для теста:   268773883692.72183
    - Значение R^2 для трейна:  0.5658247723744737
    - Значение R^2 для теста:   0.5324277455315298
    - При регуляризации занулились веса следующих признаков: km_driven, mileage, engine, seats
      
- Перебором по сетке (c 10-ю фолдами) подобрали оптимальные параметры для ElasticNet-регрессии:
    - Получили: alpha = 0.89, l1_ratio = 0.49
    - Значение MSE для трейна:  127509287985.51462
    - Значение MSE для теста:   280811983527.4693
    - Значение R^2 для трейна:  0.5551561988671906
    - Значение R^2 для теста:   0.5114856755583757


### Добавляем категориальные фичи

- Обработали поле name, оставили только марку автомобиля.
- Закодировали категориальные фичи (и seats) методом OneHot-кодирования
- В тренировочном датасете не хватает производетелей Opel и Ashok.
- Поскольку Maruti представлен больше других в обоих частях датасета, изменим две строчки на нужные нам марки
- Подобрали параметр регуляризации alpha для гребневой (ridge) регрессии:
    - Получаем: alpha = 484.1
    - Значение MSE для трейна:  65726928699.989876
    - Значение MSE для теста:   125398956846.46898
    - Значение R^2 для трейна:  0.7706973565485663
    - Значение R^2 для теста:   0.781849813102635
    - Качество предсказаний значительно улучшилось

### Решаем бизнес-задачу
- Бизнес-модель: доля предсказаний, отличающихся не более чем на 10% от реальной цены.

- Модель:  Линейная регрессия
    - Значение BM для трейна:  0.2166095890410959
    - Значение BM для теста:   0.225
      
- Модель:  Линейная регрессия со стандартизированными параметрами
    - Значение BM для трейна:  0.2166095890410959
    - Значение BM для теста:   0.225
 
- Модель:  Lasso()
    - Значение BM для трейна:  0.2166095890410959
    - Значение BM для теста:   0.225

- Модель:  Lasso(alpha=55000)
    - Значение BM для трейна:  0.20753424657534247
    - Значение BM для теста:   0.236
 
- Модель:  ElasticNet(alpha=0.89, l1_ratio=0.49)
    - Значение BM для трейна:  0.23544520547945205
    - Значение BM для теста:   0.262
 
- Модель:  Ridge(alpha=484.1)
    - Значение BM для трейна:  0.2761986301369863
    - Значение BM для теста:   0.321
 
- Лучшей с точки зрения бизнес-модели является модель Ridge(alpha=484.1), построенная на категориальных признаках, для тестовых данных 32.1% предсказаний отличаются не более 10% от реальной цены.

- Хотелось попробовать добавить в модель полиномиальных признаков, судя по графикам некоторые признаки имели схожую с квадратичной зависимость с таргетом. К сожалению не хватило времени, обязательно вернусь к этому вопросу, когда станет немного больше свободного времени.
