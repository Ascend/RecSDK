FWD_OP_DIR=$PROJECT_DIR/cust_op/ascendc_op/ai_core_op/hstu_dense_forward/v220/
BWD_OP_DIR=$PROJECT_DIR/cust_op/ascendc_op/ai_core_op/hstu_dense_backward/v220/

echo "----------------        build ops        ----------------"
rm -fr $VENDORS/hstu_dense_forward $VENDORS/hstu_dense_backward
cd $FWD_OP_DIR && sed -i '/ascend950/d' op_host/* && bash run.sh
cd $BWD_OP_DIR && sed -i '/ascend950/d' op_host/* && bash run.sh

echo "----------------        run pytest        ----------------"
cd $PRESMOKE_DIR/hstu && python3 -m pytest -x test_hstu_fwd.py test_hstu_bwd.py
