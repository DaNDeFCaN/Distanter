**Предисловие**
------
Если вы читаете это, то вы, вероятно, студент ФГРР или как то связаны с ФГРР. 
Итак, эта программа создана для оптимизации нормального радиуса пешеходной доступности, выполняя как сами расчеты, так и составляя гистограммы. 

**Режимы работы и настройка**
------
Программа запускается двумя способами: с помощью .exe файла и с помощью .py скрипта исходного кода. Первый вариант наименее требователен - не нужно ничего знать и не нужно никакой Python-среды. Второй вариант более технологичен и представляет широкие возможности к модификации и адаптации под конкретные нужды - рекомендуется всем, кто несколько знаком с Python.

**Запуск .exe**
1.	Скачать и разархивировать файл с диска и разместить его с icon.ico в одной папке. Диск: https://disk.yandex.ru/d/-PMv0VDSv77-LQ Пароль от архива: 12345
2.	Часто на файл реагирует антивирус - лучше отключить его. Подробнее в Ошибках и недочётах.
3.	Запустить приложение.
4.	Выбрать файлы .csv. Эти файлы должны соответствовать примерам.
5.	Выбрать тип соцучреждения. Доступны расчёты для детсадов и школ.
6.	Выбрать режим:

  •	Базовый расчёт: режим создаёт таблицы и гистограммы для введённого пользователем R.

  •	Оптимизация радиуса: режим производит поиск оптимального R в радиусе 400 итераций вокруг введённого пользователем R. Операция выводит промежуточные оптимумы в окно бегущей строки. Анализ и рассмотрение этих оптимумов может иметь смысл (впрочем, может и не иметь). Рекомендую после того, как нашёлся один конечный оптимум, попробовать ввести большие R и найти ещё один конечный оптимум - обычно он тоже интересен.

7.	Внести значение R. Некорректные данные, положим, отрицательный R, это ваши неверные результаты. :)
8.	Нажать старт, наслаждаться, и не бояться экспериментировать. Надпись "не отвечает" игнорировать, ждать конца расчёта, не закрывая приложение. Подробнее в Ошибках и недочётах.

**Запуск .py**
1.	Скачать файл .py и разместить его с icon.ico в одной папке.
2.	Открыть и запустить в удобной вам среде. Допустим PyCharm или Visual Studio.
3.	Повторяем шаги 4-7 алгоритма для .exe
Дополнительные возможности:
1.	Увеличить число итераций поиска. Это расширит интервал поиска, но кратно увеличит время расчёта. За это отвечает переменная iteration.
2.	Найти и раскомментировать создание таблиц под графиками. Этот блок мало тестировался и выглядит некрасиво, но удобен для проверки гистограмм - по сути по ним строятся гистограммы.
3.	Настроить вид гистограмм - можно изменить цвет, добавить обводку, изменить ширину бинов, перенастроить условие группировки по Стерджису - по умолчанию с 8 различных значений, но можно настроить и с 6, и с 9, и любого числа. Полное пространство для творчества.
4.	Изменять нормативные значения в любых пределах - так, программа может рассчитать в таком режиме радиусы для чего угодно - были бы данные и ваше стойкое желание.
5.	Менять положение плашки с асимметрией и эксцессом или вовсе удалить её.
6.	Наконец, можно как угодно ещё изменять программу, меняя отдельные функции и блоки кода - кто вас остановит теперь?


**Ошибки и недочеты**
------
1.	Наиболее часто встречающаяся проблема - блокировка антивирусом. Это неизбежно, так как: само приложение не имеет подписи, обращается к Python, установлено из неизвестного источника, работает с файловой системой - то есть выглядит крайне подозрительно, увы. Я такой же смертный, как и все, и просто не нашёл решения, которое бы устроило меня. Способы решения: игнорировать проблему и добавлять папку с .exe в исключения, восстанавливать её из карантина или отключать антивирус - по вашей ситуации ИЛИ найти добровольца с ФКН (лучше вообще кибербезопасника, это их профиль), который готов попробовать решить эту проблему ИЛИ начать долгий процесс запроса бесплатной подписи opensourse - это муторно, но я готов поспособствовать (шансов мало, честно). Проверка на Virustotal: https://www.virustotal.com/gui/file/eea5a93f63289fbeb94b4e2ab27c74e0ca2d1d8f716d9e7c95a8d8d5d75d70c9/detection
  
2.	Совершенно ложная информация о радиусе проверки в окне бегущей строки. Когда программа пишет, что она ищет в этом радиусе, она врет. Это совсем лёгкий баг графического отображения. Способы решения: игнорировать это, ведь и так ясно, что 400 итераций по 0,5 метров это не то, что там написано ИЛИ найти этот баг в коде, и исправить его, после чего написать мне, я закоммитчу это на гит и скажу спасибо! (Не только я, но и потомки;)
  
3.	Зависание программы, долгое выполнение, надпись "не отвечает" — это плохая оптимизация, господа и дамы. Создатель этого скрипта не студент ФКН, и даже не учиться на ИАДе, и писал это руками при поддержке Deepseeka в очень сжатые сроки. Программа эта - 42 часа до релиза последнего билда, третьего по счету. Не успел. На оптимизацию не осталось ни сил, ни времени. Способы решения: игнорировать это, ведь программа это все ещё лучше чем 2 человеко-часа на одну таблицу ИЛИ найти студента ФКН, что это починит ИЛИ самим пойти и сделать это, нейросети в помощь.


**Формат CSV и расшифровка таблиц**
------
Раздедитель .csv файлов - запятая, на всех тестах при стандартной выгрузке из QGIS. За чтение отвечает библиотека pandas, и проблем быть не должно, но если что-то не работает, а в вашем .csv вы не уверены, то лучше бы проверить его. Сообщения о потери файла, ошибки 'population' и 'capacity' обыкновенно связаны с тем, что вы неверно создали .csv или, что чаще, просто перепутали их.

| Имя столбца в файле выгрузки  | Имя переменной по методологии |
| ------------- | ------------- |
| Для файла домов: |
| idkindgartn  | ID учереждений дошкольного образования  |
| propdostID  | Население жилого дома распределенное пропорционально емкости доступного объекта для ID = "ID"  |
| propobesID  | Население жилого дома распределенное пропорциональноемкости доступного объекта ID = "ID"  |
| obespech  | Обеспеченность жителей дома  |
| Для файла соцучреждения: |
| sumdost  | Распределенная нагрузка на объект от всех претендентов по емкости  |
| depand  | Количество претендентов на место (спрос)  |
| 1depand  | Обеспеченность спроса на место  |
| sumobes  | Распределенная нагрузка на объект от всех претендентов по обеспеченности спроса  |
| srednagr  | Нагрузка на объект взятая как среднее между LYe и LZe  |
| obes  | Обеспеченность претендентов в каждом доступном объекте  |

**Технические ньюансы**
------
Алгоритм задействует некоторые статистические методы сверх обычной методологии.
Правило Стерджиса для гистограмм - начиная с 8 уникальных значений obespech гистограмма строится для сгруппированых по правилу данных с целью улучшения графического отображения приблежения к нормальнону распределению. Для таблиц с меньшим числом уникальных значений obespech гистограмма строится для уникальных значений с установленной шириной столбиков.
Критерий Шапиро-Уилка для оптимизации - один из наиболее чувствительных и легкореализуемых методов для численного отображения сходства распределения с нормальным распределением.


Методология из исследования-ВКР О.В. Пономаренко "Принципы оценки соответствия территорий города нормативам обеспеченности населения объектами социальной инфраструктуры" адаптированная для семинарской работы.
Ссылка: https://www.hse.ru/edu/vkr/206732273

**Идеи и предложения**
------
Что можно еще реализовать? Очень много, больше идей, чем результатов.
1. Исправление ошибок и недочетов - самое очевидное.
2. Вывод в явной форме 7 шага для домов - сейчас этот расчет "в ящике", но можно исправить это.
3. Создание раскрывающегося на стрелочку дополнительного меню с настройкой числа итераций и границ применимости правила Стерджиса.
4. Применения нового критерия нормальности. Критерий Шапиро-Уилка вполне вероятно, слишком чувствителен. Можно реализовать вместо него гораздо более сбалансированный критерий Колмогорова-Смирнова, или вообще подойти с другой стороны и взять критерий Муроты-Такеучи как наименее мощный, но вполне годный в разрезе дифференциорованых малых выборок. Или критерий Андерсона-Дарлинга, как самый популярный. Так или иначе, это, мне кажется, может быть отличным дополнением к новой работе.
5. Совсем безумная идея: сопрячь эту программу с расчетом населения или даже с автопарсером с данных ЖКХ (или иной базой данных), что позволило бы быстро рассчитывать огромные пространства.
6. Расчет дистанции сейчас происходит по прямой самым простым алгебраическим методом по прямой, без учета изгиба поверхности земли, что возможное допущение. Но можно это исправить, добавив Python-библиотеки QGISа, и при должном старании, даже создать вариант расчета для изохрон.
7. Переписать алгоритм поиска оптимального значения.
8. Написать режим поиска оптимальных расположений детского сада.
