#include <termio.h>

int getch(void)
{
	struct termios tm, tm_old;
	int fd = 0, ch;
	if(tcgetattr(fd, &tm) < 0)
	{
		return -1;
	}
	tm_old = tm;
	cfmakeraw(&tm);
	if(tcsetattr(fd, TCSANOW, &tm) < 0)
	{
		return -1;
	}
	ch = getchar();
	if(tcsetattr(fd, TCSANOW, &tm_old) < 0)
	{
		return -1;
	}
	return ch;
}
         
