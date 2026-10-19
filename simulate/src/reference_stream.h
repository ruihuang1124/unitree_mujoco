// Bounded asynchronous binary sink. Enabled only by GO2_REFERENCE_DIR.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <stdexcept>
namespace reference {
inline std::string directory() { const char* p=std::getenv("GO2_REFERENCE_DIR"); return p ? p : ""; }
inline double wallSeconds() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
class Sink {
 public:
  explicit Sink(const std::string& name) {
    const auto dir=directory(); if(dir.empty()) return;
    std::filesystem::create_directories(dir);
    path_=dir+"/"+name; file_.open(path_,std::ios::binary);
    if(!file_) throw std::runtime_error("Cannot open reference stream: "+path_);
    enabled_=true;
    worker_=std::thread([this]{
      for(;;) {
        std::vector<char> row;
        { std::unique_lock<std::mutex> lock(mutex_); cv_.wait(lock,[this]{return stop_||!queue_.empty();});
          if(queue_.empty()&&stop_) break;
          row=std::move(queue_.front()); queue_.pop_front(); }
        file_.write(row.data(),row.size()); file_.flush();
        if(!file_) failed_=true;
        std::ofstream status(path_+".status"); status<<dropped_.load()<<" "<<failed_.load()<<"\n";
      }
    });
  }
  ~Sink() { if(!enabled_)return; {std::lock_guard<std::mutex> lock(mutex_);stop_=true;}cv_.notify_all();worker_.join(); }
  bool enabled() const{return enabled_;}
  void bytes(const void* p,size_t n) {
    if(!enabled_)return;
    { std::lock_guard<std::mutex> lock(mutex_);
      if(queue_.size()>=1024){++dropped_;return;}
      const char* c=static_cast<const char*>(p);queue_.emplace_back(c,c+n); }
    cv_.notify_one();
  }
  void row(const std::vector<double>& v){bytes(v.data(),v.size()*sizeof(double));}
  void text(const std::string& v){bytes(v.data(),v.size());}
 private:
  bool enabled_=false,stop_=false; std::atomic<size_t> dropped_{0};std::atomic<bool> failed_{false};
  std::string path_;std::ofstream file_;std::mutex mutex_;std::condition_variable cv_;
  std::deque<std::vector<char>> queue_;std::thread worker_;
};
}
