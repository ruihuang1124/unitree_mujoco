#pragma once
#include "reference_stream.h"
#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>
#include <array>
#include <cmath>
namespace reference {
// Protected by the simulator mutex whenever recording is enabled.
inline std::array<double,100> motor_commands{}; // SDK order: q,dq,kp,kd,tau_ff per motor
inline double command_sim_time=-1,command_wall_time=-1;
inline bool startup_hold=false;
inline void capture(const mjModel* m,const mjData* d) {
 static Sink sink("sim.bin");if(!sink.enabled())return;
 static bool initialized=false;static double next=0,last=-1;static unsigned long step=0,reset=0;
 constexpr int max_contacts=64,contact_width=21;
 if(d->time<last){next=0;++reset;}last=d->time;++step;
 if(d->time+1e-9<next)return;next=(std::floor((d->time+1e-9)*100)+1)/100;
 if(!initialized){
  YAML::Node meta;meta["schema_version"]=1;meta["mujoco_version"]=mj_versionString();
  meta["timestep"]=m->opt.timestep;meta["max_contacts"]=max_contacts;
  meta["contact_width"]=contact_width;meta["dtype"]="little-endian float64";
  for(const auto& pair:std::vector<std::pair<std::string,int>>{{"nq",m->nq},{"nv",m->nv},{"nu",m->nu},{"na",m->na},{"nbody",m->nbody},{"nmocap",m->nmocap},{"neq",m->neq},{"nuserdata",m->nuserdata},{"nsensordata",m->nsensordata}})meta[pair.first]=pair.second;
  std::ofstream out(directory()+"/sim_schema.yaml");out<<meta;
  initialized=true;
 }
 std::vector<double> v;v.reserve(2500);
 auto add=[&](const auto* p,int n){for(int i=0;i<n;++i)v.push_back(p[i]);};
 v={d->time,wallSeconds(),double(step),double(reset),command_sim_time,command_wall_time,double(startup_hold),double(d->ncon)};
 add(d->qpos,m->nq);add(d->qvel,m->nv);add(d->act,m->na);add(d->ctrl,m->nu);
 add(d->actuator_force,m->nu);add(d->qfrc_actuator,m->nv);
 add(d->mocap_pos,3*m->nmocap);add(d->mocap_quat,4*m->nmocap);add(d->eq_active,m->neq);
 add(d->qfrc_applied,m->nv);add(d->xfrc_applied,6*m->nbody);add(d->userdata,m->nuserdata);
 add(d->sensordata,m->nsensordata);add(motor_commands.data(),100);
 for(int i=0;i<max_contacts;++i){
  if(i>=d->ncon){v.insert(v.end(),contact_width,0.);continue;}
  const auto& c=d->contact[i];v.push_back(c.geom1);v.push_back(c.geom2);v.push_back(c.dist);
  add(c.pos,3);add(c.frame,9);mjtNum force[6]{};mj_contactForce(m,d,i,force);add(force,6);
 }
 sink.row(v);
}
}
