#ifndef _CLOCKWORK_API_API_COMMON_H_
#define _CLOCKWORK_API_API_COMMON_H_

#include <functional>
#include <string>

const int clockworkSuccess = 0;
const int clockworkError = 1;
const int clockworkInitializing = 2;
const int clockworkInvalidRequest = 2;

namespace clockwork {

struct RequestHeader {
	int user_id;
	int user_request_id;
};

struct ResponseHeader {
	int user_request_id;
	int status;
	std::string message;
};

}

#endif