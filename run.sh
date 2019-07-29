#!/bin/bash

case $1 in
    train)
        python src/train.py \
            --emb_dir $(pwd)/emb/helmet \
        ;;

    test)
        python src/test.py \
            --emb_dir $(pwd)/emb/lfw \
        ;;
    *)
		echo "illegal argument"
		exit 1
    ;;
esac
exit 0