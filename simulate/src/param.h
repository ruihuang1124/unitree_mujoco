#pragma once

#include <iostream>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace param
{

inline struct SimulationConfig
{
    std::string robot;
    std::filesystem::path robot_scene;

    int domain_id;
    std::string interface;

    int use_joystick;
    std::string joystick_type;
    std::string joystick_device;
    int joystick_bits;

    int print_scene_information;

    int enable_elastic_band;
    int band_attached_link = 0;
    bool record_mocap_pair = false;
    std::string mocap_pair_csv = "/tmp/mujoco_pair_go2_base_piper_ee.csv";
    double mocap_pair_rate = 50.0;
    bool debug_rviz = false;
    std::string debug_rviz_udp_host = "127.0.0.1";
    int debug_rviz_udp_port = 16001;
    double debug_rviz_rate = 50.0;

    void load_from_yaml(const std::string &filename)
    {
        auto cfg = YAML::LoadFile(filename);
        try
        {
            robot = cfg["robot"].as<std::string>();
            robot_scene = cfg["robot_scene"].as<std::string>();
            domain_id = cfg["domain_id"].as<int>();
            interface = cfg["interface"].as<std::string>();
            use_joystick = cfg["use_joystick"].as<int>();
            joystick_type = cfg["joystick_type"].as<std::string>();
            joystick_device = cfg["joystick_device"].as<std::string>();
            joystick_bits = cfg["joystick_bits"].as<int>();
            print_scene_information = cfg["print_scene_information"].as<int>();
            enable_elastic_band = cfg["enable_elastic_band"].as<int>();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }
} config;

/* ---------- Command Line Parameters ---------- */
namespace po = boost::program_options;

//※ This function must be called at the beginning of main() function
inline po::variables_map helper(int argc, char** argv)
{
    po::options_description desc("Unitree Mujoco");
    desc.add_options()
        ("help,h", "Show help message")
        ("domain_id,i", po::value<int>(&config.domain_id), "DDS domain ID; -i 0")
        ("network,n", po::value<std::string>(&config.interface), "DDS network interface; -n eth0")
        ("robot,r", po::value<std::string>(&config.robot), "Robot type; -r go2")
        ("scene,s", po::value<std::filesystem::path>(&config.robot_scene), "Robot scene file; -s scene_terrain.xml")
        ("record_mocap_pair", po::bool_switch(&config.record_mocap_pair), "Record base_link->piper_ee CSV for mocap comparison")
        ("mocap_pair_csv", po::value<std::string>(&config.mocap_pair_csv), "Output CSV path for --record_mocap_pair")
        ("mocap_pair_rate", po::value<double>(&config.mocap_pair_rate), "CSV record rate in Hz for --record_mocap_pair")
        ("debug_rviz", po::bool_switch(&config.debug_rviz), "Publish MuJoCo world/base/EE frames over UDP for RViz")
        ("debug_rviz_udp_host", po::value<std::string>(&config.debug_rviz_udp_host), "UDP host for --debug_rviz")
        ("debug_rviz_udp_port", po::value<int>(&config.debug_rviz_udp_port), "UDP port for --debug_rviz")
        ("debug_rviz_rate", po::value<double>(&config.debug_rviz_rate), "Publish rate in Hz for --debug_rviz")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(0);
    }

    return vm;
}

}
