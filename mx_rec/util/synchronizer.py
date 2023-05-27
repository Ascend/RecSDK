#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
import socket
from time import sleep

from mx_rec.util.initialize import get_rank_id


class Communicator:
    def __init__(self):
        self.socket = socket.socket()
        self.host = socket.gethostname()
        logging.debug(f"host: {self.host}")
        self.port = 12345
        self.rank_id = get_rank_id()
        self.local_rank_id = self.rank_id % 8
        self.build_connection()

    def build_connection(self):
        if self.local_rank_id == 0:
            self.socket.bind((self.host, self.port))
            self.socket.listen(8)

        else:
            i = 0
            while True:
                try:
                    self.socket.connect((self.host, self.port))
                    break
                except ConnectionRefusedError:
                    sleep(0.01)

                i += 1
                logging.debug(f"Connection failed at the NO.{i} time for local rank id {self.local_rank_id}, "
                             f"rank id {self.rank_id}")
                if i > 200:
                    raise EnvironmentError(f"Socket connecting over time.")

            logging.debug(f"Connection was build for local rank id {self.local_rank_id}, rank id {self.rank_id}")


    def server_reply(self):
        conn, address = self.socket.accept()
        client_data = conn.recv(1024).decode()
        logging.debug(f"connecting address：{address}")
        logging.debug(f"Receive client msg: {client_data}")
        conn.send(b"Acknowledged!")
        conn.close()
        return client_data

    def client_connect(self):
        info = str(self.local_rank_id).encode()
        self.socket.send(info)
        server_reply = self.socket.recv(1024).decode()
        if server_reply != "Acknowledged!":
            raise IOError("Got a unexpected string.")

        logging.debug(f"Got the reply from local rank 0 for local rank id {self.local_rank_id}, "
                      f"rank id {self.rank_id}.")

        self.socket.close()


if __name__ == "__main__":
    communicator = Communicator()
    if communicator.local_rank_id != 0:
        communicator.client_connect()

    else:
        synchronizer_check_list = [i for i in range(1, 8)]
        while synchronizer_check_list:
            idx = int(communicator.server_reply())
            synchronizer_check_list.remove(idx)
            logging.info(f"Remove NO.{idx} element for synchronizer_check_list.")

        logging.info(f"Saver synchronized.")
