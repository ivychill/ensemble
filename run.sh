#!/bin/bash

case $1 in
    train)
        python src/train.py \
            --emb_dir $(pwd)/emb/lfw \
        ;;

    test)
        python src/test.py \
            --emb_dir $(pwd)/emb/camera \
        ;;
    *)
		echo "illegal argument"
		exit 1
    ;;
esac
exit 0