# insightface preprocessing

## بخش اول: برش زدن صورت افراد
این کتابخانه جهت پیش پردازش داده های تشخیص چهره طراحی شده است. . ابتدا برای اینکه تصاویر موجود در دیتاست را clean کنیم، فایل preprocess.py را اجرا میکنیم. این clean کردن شامل برش صورت های افراد، تشخیص نزدیک ترین فرد در هر تصویر به فرد موجود در کلاس،‌تشخیص وجود موانع در برابر صورت مثل عینک دودی و ماسک و همچنین جداسازی تصاویر بی کیفیت از نظر نور و sharpness میباشد. آدرس فایل دیتاست در dataset_path مشخص می شود،‌همچنین دیتای clean در دایرکتوری dataset_output و دیتای غیر clean در دایرکتوری dataset_temp ریخته میشود.

دو فایل preprocess.py و preprocess2.py جهت برش دادن صورت هر فرد در تصاویر آن فرد طراحی شده اند با این تفاوت که در preprocess2 امکان به کار گیری مدل هاface detector و face matcher مختلف فراهم شده است.(TODO: merge preprocess2 and preprocess).

## بخش دوم: آماده سازی داده آموزش
پس از به کارگیری فایلهای بالا و آماده سازی داده به صورت دستی داده ها را به دو بخش تست و آموزش تقسیم کنید و در فولدر valid و train قرار دهید .(TODO: automate this process) همچنین داده ولیدیشن دیگر مثل lfw.bin را نیز در این پوشه قرار میدهیم.


حال داده آموزش آماده است و تنها بایستی اندیس تصاویر را با استفاده از دستور زیر تولید کنید:

```shell
python -m mxnet.tools.im2rec --list --recursive train_output train
python -m mxnet.tools.im2rec --num-thread 16 --quality 100 train_output train
```
سپس اسم فایل ها train_output با فرمت lst rec , idx را به train تغییر میدهیم. مانند شکل زیر:
![image](./figures/Screenshot%20from%202022-05-16%2011-10-05.png)
## بخش سوم: آماده سازی داده تست


۱. داده های ولیدیشن را مطابق با lfw نامگذاری می کنیم تا بتوان داده .bin آن را تولید کرد. برای این کار فایل renamer.py را اجرا می کنیم. در این فایل نیاز است که در پارامتر root آدرس پوشه validation را بدهیم.

۲. حال با استفاده از lfw_pair_gen.py جفت داده های pair و غیر pair را تولید میکنیم. ورودی های parser دو متغییر data-dir و txt-file هستند که به ترتیب آدرس فایل valid و ادرس فایل با اسم pairs.txt هستند. اگر پوشه ای خالی بود آن را دستی حذف کنید!! (نیاز به تغییر کد)

```shell
python lfw_pair_gen.py --data-dir dataset_112*112_cleaned_2/valid/ --txt-file ./pairs.txt
```


۳. حال در این مرحله فایل pairs.txt را که در مرحله قبل تولید کردید، به پوشه valid منتقل کنید و پس اجرای دستور زیر مجدد آن را به جای اولش بازگردانید.
```shell
python dataset2bin.py --data-dir dataset_112*112_cleaned_2/valid/ --output dataset_112*112_cleaned_2/valid.bin
```



دیتاست در دایرکتوری dataset_112*112_clean_2  آماده است.