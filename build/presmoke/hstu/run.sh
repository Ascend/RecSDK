OP_DIR=$PROJECT_DIR/cust_op/ascendc_op/ai_core_op/hstu_dense_forward/v220/

echo "----------------        build ops        ----------------"
rm -fr $VENDORS/hstu_dense_forward
cd $OP_DIR && sed -i '/ascend950/d' op_host/* && bash run.sh

echo "----------------        run pytest        ----------------"
cd $PRESMOKE_DIR/hstu && python3 -m pytest -x test_hstu_fwd.py
