#ifndef DYNAMIXEL_CLIENT_H
#define DYNAMIXEL_CLIENT_H

#include <dynamixel_sdk.h>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <mutex>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class DynamixelClient
{
public:
  static constexpr float PROTOCOL_VERSION = 2.0f;
  static constexpr float DEFAULT_POS_SCALE = 2.0f * M_PI / 4096.0f;
  static constexpr float DEFAULT_VEL_SCALE = 0.229f * 2.0f * M_PI / 60.0f;
  static constexpr float DEFAULT_V_IN_SCALE = 0.1f;
  static constexpr float DEFAULT_PWM_SCALE = 0.113f;
  static constexpr float DEFAULT_TEMP_SCALE = 1.0f;

  // Control table addresses
  static constexpr uint16_t ADDR_MODEL_NUMBER = 0;
  static constexpr uint16_t ADDR_TORQUE_ENABLE = 64;
  static constexpr uint16_t ADDR_GOAL_PWM = 100;
  static constexpr uint16_t ADDR_GOAL_CURRENT = 102;
  static constexpr uint16_t ADDR_GOAL_VELOCITY = 104;
  static constexpr uint16_t ADDR_GOAL_POSITION = 116;
  static constexpr uint16_t ADDR_PRESENT_VELOCITY = 128;
  static constexpr uint16_t ADDR_PRESENT_CURRENT = 126;
  static constexpr uint16_t ADDR_PRESENT_POS_VEL_CUR = 126;
  static constexpr uint16_t ADDR_PRESENT_POSITION = 132;
  static constexpr uint16_t ADDR_PRESENT_V_IN = 144;
  static constexpr uint16_t ADDR_PRESENT_TEMP = 146;
  static constexpr uint16_t ADDR_PRESENT_PWM = 124;
  static constexpr uint16_t ADDR_PRESENT_ALL = 124;
  static constexpr uint16_t ADDR_CURRENT_LIMIT = 38;

  // Data lengths
  static constexpr uint16_t LEN_MODEL_NUMBER = 2;
  static constexpr uint16_t LEN_GOAL_PWM = 2;
  static constexpr uint16_t LEN_GOAL_CURRENT = 2;
  static constexpr uint16_t LEN_GOAL_VELOCITY = 4;
  static constexpr uint16_t LEN_GOAL_POSITION = 4;
  static constexpr uint16_t LEN_PRESENT_CURRENT = 2;
  static constexpr uint16_t LEN_PRESENT_VELOCITY = 4;
  static constexpr uint16_t LEN_PRESENT_POSITION = 4;
  static constexpr uint16_t LEN_PRESENT_POS_VEL_CUR = 10;
  static constexpr uint16_t LEN_PRESENT_V_IN = 2;
  static constexpr uint16_t LEN_PRESENT_TEMP = 1;
  static constexpr uint16_t LEN_PRESENT_PWM = 2;
  static constexpr uint16_t LEN_PRESENT_ALL = 23;
  static constexpr uint16_t LEN_CURRENT_LIMIT = 2;

  // ctor/dtor
  DynamixelClient(const std::vector<int> &motor_ids,
                  const std::string &port = "/dev/ttyUSB0",
                  int baudrate = 1000000,
                  bool lazy_connect = false);
  ~DynamixelClient();

  // connection
  void connect();
  void disconnect();
  bool is_connected() const;

  // torque & reboot
  void set_torque_enabled(const std::vector<int> &ids,
                          bool en,
                          int retries = -1,
                          float retry_interval = 0.25f);
  void reboot(const std::vector<int> &ids);
  void clear_multi_turn(const std::vector<int> &ids);
  void clear_error(const std::vector<int> &ids);

  // high-level reads
  std::pair<double, std::vector<float>> read_model_number(int retries = 0);
  std::pair<double, std::vector<float>> read_pos(int retries = 0);
  std::pair<double, std::vector<float>> read_vel(int retries = 0);
  std::pair<double, std::vector<float>> read_cur(int retries = 0);
  std::pair<double, std::vector<float>> read_cur_limit(int retries = 0);
  std::pair<double, std::vector<float>> read_vin(int retries = 0);
  std::tuple<double, std::vector<float>, std::vector<float>>
  read_pos_vel(int retries = 0);
  std::tuple<double, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>>
  read_all(int retries = 0);

  // high-level writes
  void write_desired_pos(const std::vector<int> &ids,
                         const std::vector<float> &positions);
  void write_desired_vel(const std::vector<int> &ids,
                         const std::vector<float> &velocities);
  void write_desired_pwm(const std::vector<int> &ids,
                         const std::vector<float> &pwm);
  void write_desired_cur(const std::vector<int> &ids,
                         const std::vector<float> &currents);

  // low-level grouped operations
  void sync_write(const std::vector<int> &ids,
                  const std::vector<int> &values,
                  uint16_t addr,
                  uint16_t len);
  std::pair<double, std::vector<float>>
  sync_read(uint16_t addr, uint16_t len, float scale);
  std::pair<double, std::map<std::string, std::vector<float>>>
  bulk_read(const std::vector<std::string> &attrs, int retries = 0);

  // utility
  std::vector<float> get_cur_scale(int retries = 0);

  static int unsigned_to_signed(int value, int size);
  static int signed_to_unsigned(int value, int size);

  static const std::set<DynamixelClient *> &get_open_clients()
  {
    return OPEN_CLIENTS;
  }

private:
  // get current timestamp
  inline double now();

  // internal helpers
  void check_connected();
  bool handle_packet_result(int comm,
                            int dxl_err,
                            int id,
                            const std::string &ctx);
  std::vector<int> write_byte(const std::vector<int> &ids,
                              int val,
                              uint16_t addr);

  // state
  bool is_open_ = false;
  static std::set<DynamixelClient *> OPEN_CLIENTS;
  static std::once_flag cleanup_flag;

  std::vector<int> motor_ids_;
  std::string port_name_;
  int baudrate_;
  bool lazy_connect_;

  // SDK handlers
  dynamixel::PortHandler *port_handler_;
  dynamixel::PacketHandler *packet_handler_;
  dynamixel::GroupBulkRead bulk_reader_;

  std::map<std::pair<uint16_t, uint16_t>, dynamixel::GroupSyncRead *> sync_readers_;
  std::map<std::pair<uint16_t, uint16_t>, dynamixel::GroupSyncWrite *> sync_writers_;

  std::map<std::string, std::vector<float>> data_dict_;
  std::map<std::string, std::vector<float>> last_bulk_res_;
  std::vector<float> cur_scale_arr_;

  std::mutex comms_mutex_;
};

void dynamixel_cleanup_handler();

#endif // DYNAMIXEL_CLIENT_H
