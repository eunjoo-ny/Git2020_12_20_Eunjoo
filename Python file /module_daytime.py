{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "output the current time\n",
      "2020 year\n",
      "9 month\n",
      "5 day\n",
      "17 hour\n",
      "1 minute\n",
      "13 second\n",
      "\n",
      "output the current time considering format\n",
      "2020.09.09/05/20 17:01:13\n",
      "2020year 9month 5day 17hour 1minute 13second\n",
      "2020y 09m 09/05/20d 17h 01m 13s\n"
     ]
    }
   ],
   "source": [
    "import datetime\n",
    "\n",
    "now=datetime.datetime.now()\n",
    "print(\"output the current time\")\n",
    "print(now.year,\"year\")\n",
    "print(now.month,\"month\")\n",
    "print(now.day,\"day\")\n",
    "print(now.hour,\"hour\")\n",
    "print(now.minute,\"minute\")\n",
    "print(now.second,\"second\")\n",
    "print()\n",
    "        \n",
    "\n",
    "print(\"output the current time considering format\")\n",
    "output_a=now.strftime(\"%Y.%m.%D %H:%M:%S\")\n",
    "output_b=\"{}year {}month {}day {}hour {}minute {}second\".format(now.year,\\\n",
    "                                                                now.month,\\\n",
    "                                                                now.day,\\\n",
    "                                                                now.hour,\\\n",
    "                                                                now.minute,\\\n",
    "                                                                now.second)\n",
    "output_c=now.strftime(\"%Y{} %m{} %D{} %H{} %M{} %S{}\").format(*\"ymdhms\")\n",
    "print(output_a)\n",
    "print(output_b)\n",
    "print(output_c)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
