#ifndef QUATRBT_H
#define QUATRBT_H

#include <iostream>
#include <cassert>
#include <cmath>

#include "matrix4.h"
#include "cvec.h"
#include "quat.h"

class QuatRBT {
  Cvec3 t_; // translation component
  Quat r_;  // rotation component represented as a quaternion

public:
  QuatRBT() : t_(0) {
    assert(norm2(Quat(1,0,0,0) - r_) < CS175_EPS2);
  }

  QuatRBT(const Cvec3& t, const Quat& r) {
    t_ = t;
    r_ = r;
  }

  explicit QuatRBT(const Cvec3& t) {
    // TODO
    t_ = t;
    r_ = Quat(1,0,0,0);
  }

  explicit QuatRBT(const Quat& r) {
    // TODO
    r_ = r;
    t_ = Cvec3(0,0,0);
  }

  Cvec3 getTranslation() const {
    return t_;
  }

  Quat getRotation() const {
    return r_;
  }

  QuatRBT& setTranslation(const Cvec3& t) {
    t_ = t;
    return *this;
  }

  QuatRBT& setRotation(const Quat& r) {
    r_ = r;
    return *this;
  }

  Cvec4 operator * (const Cvec4& a) const {
    // Rob
    return (r_ * a + Cvec4(t_, 0));
  }

  QuatRBT operator * (const QuatRBT& a) const {
    // Rob
    //does the opperations listen on page 70 of graphics book
    //also converts to Cvec4, and theb back to cvec3
    Cvec4 tran = Cvec4(t_,0)  + (r_ * Cvec4(a.getTranslation(),0));
    Cvec3 trans = Cvec3(tran(0), tran(1), tran(2));
    Quat rot = r_ * a.getRotation();
    QuatRBT ans = QuatRBT();
    ans.setTranslation(trans);
    ans.setRotation(rot);
    return ans;
  }
};

inline QuatRBT inv(const QuatRBT& tform) {
  // TODO
  QuatRBT rt = QuatRBT();
  Quat invr = inv(tform.getRotation());
  rt.setRotation(invr);
  Cvec4 trns = invr * Cvec4(tform.getTranslation(), 0) * (-1.0);
  rt.setTranslation(Cvec3(trns(0), trns(1), trns(2)));
  return rt;
}

inline QuatRBT transFact(const QuatRBT& tform) {
  //Rob 
  return QuatRBT(tform.getTranslation());
}

inline QuatRBT linFact(const QuatRBT& tform) {
  // Rob
  return QuatRBT(tform.getRotation());
}

inline Matrix4 rigTFormToMatrix(const QuatRBT& tform) {
  // TODO
  Matrix4 m = quatToMatrix(tform.getRotation());
  Cvec3 t = tform.getTranslation();
  m(0,3) = t(0);
  m(1,3) = t(1);
  m(2,3) = t(2);
  return m;
}

#endif