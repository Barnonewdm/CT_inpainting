#!/bin/bash
SECONDS=0
while getopts "t:s:" OPT
  do
  case $OPT in
      t)
   tem=$OPTARG
   ;;
      s)
   seg=$OPTARG
   ;;
     \?)
   echo "$USAVE">&2
   exit 1
   ;;
  esac
done
echo "dong process of $tem file for inpainting"
python process_for_test.py --input_ori_name $tem --input_mask_name $seg
for img in ./data/liver_image/test/predicted/img/test_*
do
	file=${img#*test_}
	base=${img%test_*}
	mv $img $base/predicted_$file
done
echo "doing inpainting slice by slice"
:<<BLOCK
for img in ./data/liver_image/test/ct/mask/*
do
	seg=${img#*seg_}
	base=${img%/mask/seg_*}
	python predict.py --input_ori_name $base/img/test_$seg --input_mask_name $img --save_name ./data/liver_image/test/predicted/img/predicted_$seg
	python predict.py --input_ori_name $base/img/test_$seg --input_mask_name $img --save_name ./data/liver_image/test/predicted/img/predicted_ --input_dir ./data/liver_image/test/ct/
done
BLOCK
python predict.py --input_ori_name ./data/liver_image/test/ct/img/test_50.png --input_mask_name ./data/liver_image/test/ct/mask/seg_50.png --save_name ./data/liver_image/test/predicted/img/predicted_ --input_dir ./data/liver_image/test/ct/
echo "doing after_process to stack 3D image"
save_name=${tem##*/}
save_name=${save_name%.nii*}
python after_process.py --tem $tem --data_dir ./data/liver_image/test/predicted/img/ --save_name ./data/liver_image/test/predicted/${save_name}_inpainted.nii
rm ./data/liver_image/test/ct/img/*
rm ./data/liver_image/test/ct/mask/*
echo "[finished]: Totally $SECONDS secs"
