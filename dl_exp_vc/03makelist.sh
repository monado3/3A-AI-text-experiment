#!/bin/bash

if [ ! -e conf ];then
	mkdir conf
fi

ls data/SF/data/ | head -45 | sed -e 's/\.dat//' > conf/train.list
ls data/SF/data/ | tail -5 | sed -e 's/\.dat//' > conf/eval.list
