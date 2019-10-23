#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

CXX = c++
CXXFLAGS = -pthread -march=native -fPIC -std=c++11 # -flto=full
OBJS = args.o autotune.o matrix.o dictionary.o loss.o productquantizer.o densematrix.o quantmatrix.o vector.o model.o utils.o meter.o fasttext.o fasttextd.o
INCLUDES = -I.
SRC = fastText/src

opt: CXXFLAGS += -O3 -funroll-loops
opt: fasttext

coverage: CXXFLAGS += -O0 -fno-inline -fprofile-arcs --coverage
coverage: fasttext

debug: CXXFLAGS += -g -O0 -fno-inline
debug: fasttext

args.o: $(SRC)/args.cc $(SRC)/args.h
	$(CXX) $(CXXFLAGS) -c $(SRC)/args.cc

autotune.o: $(SRC)/autotune.cc $(SRC)/autotune.h
	$(CXX) $(CXXFLAGS) -c $(SRC)/autotune.cc

dictionary.o: $(SRC)/dictionary.cc $(SRC)/dictionary.h $(SRC)/args.h
	$(CXX) $(CXXFLAGS) -c $(SRC)/dictionary.cc

loss.o: $(SRC)/loss.cc $(SRC)/loss.h $(SRC)/matrix.h $(SRC)/real.h
	$(CXX) $(CXXFLAGS) -c $(SRC)/loss.cc

productquantizer.o: $(SRC)/productquantizer.cc $(SRC)/productquantizer.h $(SRC)/utils.h
	$(CXX) $(CXXFLAGS) -c $(SRC)/productquantizer.cc

densematrix.o: $(SRC)/densematrix.cc $(SRC)/densematrix.h $(SRC)/utils.h $(SRC)/matrix.h
	$(CXX) $(CXXFLAGS) -c $(SRC)/densematrix.cc

matrix.o: $(SRC)/matrix.cc $(SRC)/matrix.h $(SRC)/utils.h
	$(CXX) $(CXXFLAGS) -c $(SRC)/matrix.cc

quantmatrix.o: $(SRC)/quantmatrix.cc $(SRC)/quantmatrix.h $(SRC)/utils.h
	$(CXX) $(CXXFLAGS) -c $(SRC)/quantmatrix.cc

vector.o: $(SRC)/vector.cc $(SRC)/vector.h $(SRC)/utils.h
	$(CXX) $(CXXFLAGS) -c $(SRC)/vector.cc

model.o: $(SRC)/model.cc $(SRC)/model.h $(SRC)/args.h
	$(CXX) $(CXXFLAGS) -c $(SRC)/model.cc

utils.o: $(SRC)/utils.cc $(SRC)/utils.h
	$(CXX) $(CXXFLAGS) -c $(SRC)/utils.cc

meter.o: $(SRC)/meter.cc $(SRC)/meter.h
	$(CXX) $(CXXFLAGS) -c $(SRC)/meter.cc

fasttext.o: $(SRC)/fasttext.cc $(SRC)/*.h
	$(CXX) $(CXXFLAGS) -c $(SRC)/fasttext.cc

fasttextd.o: fasttextd.cc $(SRC)/*.h
	$(CXX) $(CXXFLAGS) -IfastText/src -c fasttextd.cc

static: $(OBJS)
	ar rcs libfasttext.a $^

fasttext: $(OBJS)
	$(CXX) $(CXXFLAGS) -IfastText/src -shared $(OBJS) -o libfasttext.so

combined:
	ld.gold -r $(OBJS) -o combined.o

clean:
	rm -rf *.o *.gcno *.gcda *.so *.a fasttext
