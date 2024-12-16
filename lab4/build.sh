#! /bin/bash
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

tar -xvf alsa-lib-1.0.26.tar.bz2
cd alsa-lib-1.0.26
mkdir -p "$SCRIPT_DIR/madplay"
CC=arm-linux-gnueabihf-gcc ./configure \
  --host=arm-linux-gnueabihf \
  --prefix="$SCRIPT_DIR/madplay"
make -j$(nproc)
make -j$(nproc) install
cd ..

tar -xvf alsa-utils-1.0.26.tar.bz2
cd alsa-utils-1.0.26
CC=arm-linux-gnueabihf-gcc ./configure \
  --prefix="$SCRIPT_DIR/madplay" \
  --host=arm-linux-gnueabihf \
  --with-alsa-inc-prefix="$SCRIPT_DIR/madplay/include" \
  --with-alsa-prefix="$SCRIPT_DIR/madplay/lib" \
  --disable-alsamixer \
  --disable-xmlto \
  --disable-nls
make -j$(nproc)
cd ..

tar -zxvf zlib-1.2.3.tar.gz
cd zlib-1.2.3
./configure --prefix="$SCRIPT_DIR/madplay"
sed -e 's/^CC *=.*/CC=arm-linux-gnueabihf-gcc/' \
  -e 's/^AR *=.*/AR=arm-linux-gnueabihf-ar rc/' \
  -e 's/^RANLIB *=.*/RANLIB=arm-linux-gnueabihf-ranlib/' \
	-e 's/^CFLAGS=-O3 -DUSE_MMAP/CFLAGS=-O3 -fPIC/' \
  ./Makefile > Makefile1
cp -f Makefile1 Makefile
make -j$(nproc)
make -j$(nproc) install
cd ..

tar -zxvf libid3tag-0.15.1b.tar.gz
cd libid3tag-0.15.1b
./configure --host=arm-linux-gnueabihf \
  --disable-debugging \
  --prefix="$SCRIPT_DIR/madplay" \
  CPPFLAGS=-I"$SCRIPT_DIR/madplay/include" \
  LDFLAGS=-L"$SCRIPT_DIR/madplay/lib"
make -j$(nproc)
make -j$(nproc) install
cd ..

tar -zxvf libmad-0.15.1b.tar.gz
cd libmad-0.15.1b
./configure --host=arm-linux-gnueabihf \
  --disable-debugging \
  --prefix="$SCRIPT_DIR/madplay" \
  CPPFLAGS=-I"$SCRIPT_DIR/madplay/include" \
  LDFLAGS=-L"$SCRIPT_DIR/madplay/lib"
make -j$(nproc) || true
sed '/-fforce-mem/d' ./configure > ./configure1
cp -f configure1 configure
chmod u+x configure
./configure --host=arm-linux-gnueabihf \
  --prefix=/usr/local/libmad_arm \
  --enable-shared \
  --enable-static \
  --enable-fpm=arm \
  --with-gnu-ld=arm-linux-gnueabihf-ld \
  --build=arm
cp -f $SCRIPT_DIR/fixed_new.h ./fixed.h
make -j$(nproc)
make -j$(nproc) install
cd ..

tar -zxvf madplay-0.15.2b.tar.gz
cd madplay-0.15.2b
./configure --host=arm-linux-gnueabihf \
  CC=arm-linux-gnueabihf-gcc \
  --disable-debugging \
  --with-alsa \
  CPPFLAGS="-I$SCRIPT_DIR/madplay/include -I/usr/local/libmad_arm/include" \
  LDFLAGS="-L$SCRIPT_DIR/madplay/lib -L/usr/local/libmad_arm/lib"
make -j$(nproc)
make -j$(nproc) install
