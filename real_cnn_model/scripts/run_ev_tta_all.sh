echo time
./run_ev_tta.sh reshape_then_acc_time_pol ./experiments/pretrained/2_time_surface/best.tar
echo count
./run_ev_tta.sh reshape_then_acc_count_pol ./experiments/pretrained/2_count_surface/best.tar
echo exp
./run_ev_tta.sh reshape_then_acc_exp ./experiments/pretrained/2_exp_surface/best.tar
echo dist
./run_ev_tta.sh reshape_then_acc_adj_sort ./experiments/pretrained/2_disc_sort/best.tar
echo event_img
./run_ev_tta.sh reshape_then_flat_pol ./experiments/pretrained/2_event_image/best.tar
echo sort
./run_ev_tta.sh reshape_then_acc_sort ./experiments/pretrained/2_sort_surface/best.tar
