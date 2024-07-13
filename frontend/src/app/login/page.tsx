import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import Image from "next/image";

export default function LoginPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-4 py-8">
      <div className="w-full max-w-md">
        <Image
          src="/a-star.png"
          width={160}
          height={160}
          alt="Logo"
          className="mx-auto"
        />
        <div className="mt-12">
          <h2 className="text-2xl font-semibold text-center">
            Log in to your account
          </h2>
          <p className="mt-2 text-sm text-center text-gray-600">
            Welcome back! Please enter your details
          </p>
          <form className="mt-8 space-y-6">
            <div>
              <Label className="sr-only" htmlFor="email">
                Email
              </Label>
              <Input id="email" placeholder="Email" type="email" />
            </div>
            <div>
              <Label className="sr-only" htmlFor="password">
                Password
              </Label>
              <Input id="password" placeholder="Password" type="password" />
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Checkbox id="remember-me" />
                <Label
                  className="ml-2 block text-sm text-gray-900 dark:text-white"
                  htmlFor="remember-me"
                >
                  Remember for 30 days
                </Label>
              </div>
              <Link
                className="text-sm text-gray-600 hover:text-gray-500 dark:text-white"
                href="#"
              >
                Forgot Password
              </Link>
            </div>
            <div>
              <Button className="w-full">Sign in</Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
